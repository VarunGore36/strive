mod audio;
mod worker;

use audio::{Capture, RATE};
use axum::serve::ListenerExt;
use axum::{
    Json, Router,
    extract::{
        DefaultBodyLimit, Path, State,
        ws::{Message, WebSocket, WebSocketUpgrade},
    },
    http::{HeaderValue, StatusCode},
    middleware::{self, Next},
    response::{IntoResponse, Response},
    routing::{delete, get, patch, post},
};
use futures_util::SinkExt;
use serde_json::{Value, json};
use std::{
    collections::HashMap,
    sync::{
        Arc, Mutex as StdMutex,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
    time::{Duration, Instant},
};
use tokio::{
    sync::{Mutex, Notify, OwnedSemaphorePermit, Semaphore, mpsc, watch},
    time::{sleep, timeout},
};
use tower_http::services::{ServeDir, ServeFile};
use worker::Worker;

type Error = (StatusCode, Json<Value>);
fn error(code: StatusCode, text: impl Into<String>) -> Error {
    (code, Json(json!({"detail":text.into()})))
}

struct App {
    sessions: Mutex<HashMap<String, Arc<Session>>>,
    permits: Arc<Semaphore>,
    spare: Mutex<Option<Worker>>,
    config: Value,
    public: Value,
    ready: Value,
    python: String,
    root: String,
    deadline: Duration,
    startup: Duration,
    token: String,
}
struct Session {
    id: String,
    body: Mutex<Value>,
    worker: Mutex<Option<Worker>>,
    capture: StdMutex<Capture>,
    generation: AtomicU64,
    request: AtomicU64,
    connected: AtomicBool,
    closed: watch::Sender<bool>,
    wake: Notify,
    touched: StdMutex<Instant>,
    _permit: OwnedSemaphorePermit,
}
impl Session {
    async fn stopped(&self) {
        let mut state = self.closed.subscribe();
        if *state.borrow() {
            return;
        }
        let _ = state.changed().await;
    }
    fn is_closed(&self) -> bool {
        *self.closed.borrow()
    }
    async fn rpc(&self, app: &App, mut header: Value, samples: &[f32]) -> Result<Value, String> {
        let dispatch_started = Instant::now();
        let mut slot = self.worker.lock().await;
        if self.is_closed() {
            return Err("Session closed".into());
        }
        let generation = self.generation.load(Ordering::SeqCst);
        let work = async {
            if slot.is_none() {
                let mut worker = Worker::spawn(&app.python, &app.root, app.startup).await?;
                let result = worker
                    .init(self.body.lock().await.clone(), generation, app.deadline)
                    .await?;
                if result["status"] != 201 {
                    return Err("Worker reinitialization failed".into());
                }
                *slot = Some(worker);
            }
            header["generation"] = json!(generation);
            header["request_id"] = json!(self.request.fetch_add(1, Ordering::SeqCst) + 1);
            if header["op"] == "window" {
                header["queue_ms"] = json!(
                    header["queue_ms"].as_f64().unwrap()
                        + dispatch_started.elapsed().as_secs_f64() * 1000.
                );
            }
            slot.as_mut()
                .unwrap()
                .rpc(&header, samples, app.deadline)
                .await
        };
        let result = tokio::select! { result = work => result, _ = self.stopped() => Err("Session closed".into()) };
        if result.is_err() {
            if let Some(mut worker) = slot.take() {
                worker.stop().await;
            }
            self.generation.fetch_add(1, Ordering::SeqCst);
        }
        result
    }
}

async fn lookup(app: &App, id: &str) -> Result<Arc<Session>, Error> {
    app.sessions
        .lock()
        .await
        .get(id)
        .filter(|s| !s.is_closed())
        .cloned()
        .ok_or_else(|| error(StatusCode::NOT_FOUND, "Call not found or expired"))
}

async fn remove(app: &App, session: &Session) {
    session.closed.send_replace(true);
    session.wake.notify_one();
    app.sessions.lock().await.remove(&session.id);
    if let Some(mut worker) = session.worker.lock().await.take() {
        worker.stop().await;
    }
}

async fn start(
    State(app): State<Arc<App>>,
    Json(body): Json<Value>,
) -> Result<(StatusCode, Json<Value>), Error> {
    let permit = app
        .permits
        .clone()
        .try_acquire_owned()
        .map_err(|_| error(StatusCode::TOO_MANY_REQUESTS, "Session capacity reached"))?;
    let spare = app.spare.lock().await.take();
    let mut worker = match spare {
        Some(w) => w,
        None => Worker::spawn(&app.python, &app.root, app.startup)
            .await
            .map_err(|e| error(StatusCode::SERVICE_UNAVAILABLE, e))?,
    };
    let result = worker
        .init(body.clone(), 0, app.deadline)
        .await
        .map_err(|e| error(StatusCode::SERVICE_UNAVAILABLE, e))?;
    if result["status"] != 201 {
        return Err((
            StatusCode::UNPROCESSABLE_ENTITY,
            Json(result["body"].clone()),
        ));
    }
    let id = uuid::Uuid::new_v4().simple().to_string();
    let session = Arc::new(Session {
        id: id.clone(),
        body: Mutex::new(body),
        worker: Mutex::new(Some(worker)),
        capture: StdMutex::new(Capture::new(
            samples(&app.config, "window_s"),
            samples(&app.config, "stride_s"),
            app.config["stream_queue_windows"].as_u64().unwrap() as usize,
        )),
        generation: AtomicU64::new(0),
        request: AtomicU64::new(0),
        connected: AtomicBool::new(false),
        closed: watch::channel(false).0,
        wake: Notify::new(),
        touched: StdMutex::new(Instant::now()),
        _permit: permit,
    });
    app.sessions.lock().await.insert(id.clone(), session);
    Ok((
        StatusCode::CREATED,
        Json(
            json!({"call_id":id,"ws_path":format!("/v1/stream/{id}"),"sample_rate":RATE,"encoding":"s16le"}),
        ),
    ))
}

async fn end(State(app): State<Arc<App>>, Path(id): Path<String>) -> Result<Json<Value>, Error> {
    let session = lookup(&app, &id).await?;
    remove(&app, &session).await;
    Ok(Json(
        json!({"status":"ended","ephemeral_state_deleted":true}),
    ))
}

async fn control(app: Arc<App>, id: String, action: &str, body: Value) -> Result<Response, Error> {
    let session = lookup(&app, &id).await?;
    let (sequence, received) = {
        let c = session.capture.lock().unwrap();
        (c.sequence, c.received)
    };
    let result = session.rpc(&app,json!({"op":"control","action":action,"body":body,"sequence":sequence,"received":received}),&[]).await
        .map_err(|e|error(StatusCode::SERVICE_UNAVAILABLE,e))?;
    let status = StatusCode::from_u16(result["status"].as_u64().unwrap_or(500) as u16)
        .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
    if action == "context" && status.is_success() {
        session.body.lock().await["context"] = body;
    }
    let mut response = result["body"].clone();
    if action == "audit" {
        response["call_id"] = json!(id);
    }
    Ok((status, Json(response)).into_response())
}
async fn context(
    State(a): State<Arc<App>>,
    Path(id): Path<String>,
    Json(b): Json<Value>,
) -> Result<Response, Error> {
    control(a, id, "context", b).await
}
async fn verify(
    State(a): State<Arc<App>>,
    Path(id): Path<String>,
    Json(b): Json<Value>,
) -> Result<Response, Error> {
    control(a, id, "verify", b).await
}
async fn transaction(State(a): State<Arc<App>>, Path(id): Path<String>) -> Result<Response, Error> {
    control(a, id, "transaction", Value::Null).await
}
async fn audit(State(a): State<Arc<App>>, Path(id): Path<String>) -> Result<Response, Error> {
    control(a, id, "audit", Value::Null).await
}

async fn scoring(app: Arc<App>, session: Arc<Session>, out: mpsc::Sender<Value>) {
    loop {
        if session.is_closed() {
            break;
        }
        let taken = session.capture.lock().unwrap().take();
        let Some((window, dropped)) = taken else {
            tokio::select! { _=session.wake.notified()=>{}, _=session.stopped()=>break };
            continue;
        };
        let (sequence, received, capture) = {
            let c = session.capture.lock().unwrap();
            (c.sequence, c.received, c.stats())
        };
        let header = json!({"op":"window","sequence":sequence,"received":received,
            "start":window.start,"end":window.end,"fresh":window.fresh,"capture":capture,
            "dropped":dropped,"queue_ms":window.queued.elapsed().as_secs_f64()*1000.});
        let ipc_started = Instant::now();
        let response = session.rpc(&app, header, &window.samples).await;
        if session.is_closed() {
            break;
        }
        let message = match response {
            Ok(mut result) if result["end"] == window.end => {
                let event = &mut result["event"];
                event["call_id"] = json!(session.id);
                event["runtime"] = json!("rust-python");
                event["generation"] = json!(session.generation.load(Ordering::SeqCst));
                let newest = session.capture.lock().unwrap().received;
                event["latency_ms"]["audio_lag"] =
                    json!(newest.saturating_sub(window.end) as f64 / RATE as f64 * 1000.);
                event["latency_ms"]["native_window_to_result"] =
                    json!(window.queued.elapsed().as_secs_f64() * 1000.);
                event["latency_ms"]["worker_roundtrip"] =
                    json!(ipc_started.elapsed().as_secs_f64() * 1000.);
                json!({"type":"events","events":[event],"queued":session.capture.lock().unwrap().stats()["current_depth"]})
            }
            other => {
                eprintln!(
                    "Worker unavailable for {}: {}",
                    session.id,
                    other.err().unwrap_or_else(|| "Wrong window reply".into())
                );
                json!({"type":"status","state":"ANALYZING","reason":"WORKER_UNAVAILABLE",
                    "generation":session.generation.load(Ordering::SeqCst),"message":"Detector worker unavailable; audio reception continues."})
            }
        };
        tokio::select! { r=out.send(message)=>{if r.is_err(){break;}},_=session.stopped()=>break }
    }
}

async fn send(ws: &mut WebSocket, value: Value) -> Result<(), ()> {
    timeout(
        Duration::from_secs(5),
        ws.send(Message::Text(value.to_string().into())),
    )
    .await
    .map_err(|_| ())?
    .map_err(|_| ())
}
fn authorized(expected: &str, supplied: &str) -> bool {
    let mut diff = expected.len() ^ supplied.len();
    for (a, b) in expected.bytes().zip(supplied.bytes()) {
        diff |= (a ^ b) as usize;
    }
    diff == 0
}
async fn websocket(
    State(app): State<Arc<App>>,
    Path(id): Path<String>,
    upgrade: WebSocketUpgrade,
) -> Result<Response, Error> {
    let session = lookup(&app, &id).await?;
    Ok(upgrade
        .max_message_size(3204)
        .max_frame_size(3204)
        .on_upgrade(move |socket| stream(app, session, socket)))
}
async fn stream(app: Arc<App>, session: Arc<Session>, mut ws: WebSocket) {
    let initial = timeout(Duration::from_secs(5), ws.recv()).await;
    let auth = match initial {
        Ok(Some(Ok(Message::Text(t)))) => serde_json::from_str::<Value>(&t).ok(),
        _ => None,
    };
    let allowed = auth.as_ref().is_some_and(|a| {
        authorized(&app.token, a["token"].as_str().unwrap_or("")) && a["protocol"] == "pcm-v2"
    });
    if !allowed
        || session
            .connected
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
            .is_err()
    {
        let _=send(&mut ws,json!({"type":"error","message":"Unauthorized, unsupported protocol, or active producer"})).await;
        let _ = ws.close().await;
        return;
    }
    if send(
        &mut ws,
        json!({"type":"ready","sample_rate":RATE,"protocol":"pcm-v2","frame_ms":20,
        "window_s":app.config["window_s"],"hop_s":app.config["stride_s"],"runtime":"rust-python"}),
    )
    .await
    .is_err()
    {
        remove(&app, &session).await;
        return;
    }
    let (out, mut replies) = mpsc::channel(4);
    let task = tokio::spawn(scoring(app.clone(), session.clone(), out));
    let idle = Duration::from_secs(app.config["idle_timeout_s"].as_u64().unwrap());
    loop {
        let incoming = tokio::select! {
            _=session.stopped()=>break,
            _=sleep(idle)=>break,
            value=replies.recv()=>{if let Some(v)=value {if send(&mut ws,v).await.is_err(){break;}} else {break;} continue;},
            value=ws.recv()=>value,
        };
        match incoming {
            Some(Ok(Message::Binary(data))) => {
                let started = Instant::now();
                let result = {
                    let mut c = session.capture.lock().unwrap();
                    c.ingest(
                        &data,
                        app.config["max_call_s"].as_u64().unwrap() as usize * RATE,
                    )
                    .map(|_| (c.sequence - 1, c.stats()))
                };
                match result {
                    Ok((sequence, capture)) => {
                        *session.touched.lock().unwrap() = Instant::now();
                        session.wake.notify_one();
                        if send(
                            &mut ws,
                            json!({"type":"ack","sequence":sequence,"capture":capture,
                            "ingest_ms":started.elapsed().as_secs_f64()*1000.}),
                        )
                        .await
                        .is_err()
                        {
                            break;
                        }
                    }
                    Err(e) => {
                        let _ = send(&mut ws, json!({"type":"error","message":e})).await;
                        break;
                    }
                }
            }
            Some(Ok(Message::Ping(p))) => {
                if ws.send(Message::Pong(p)).await.is_err() {
                    break;
                }
            }
            Some(Ok(Message::Close(_))) | None | Some(Err(_)) => break,
            _ => {
                let _ = send(
                    &mut ws,
                    json!({"type":"error","message":"Binary pcm-v2 required"}),
                )
                .await;
                break;
            }
        }
    }
    remove(&app, &session).await;
    let _ = task.await;
    let _ = ws.close().await;
}

async fn access(
    State(app): State<Arc<App>>,
    request: axum::extract::Request,
    next: Next,
) -> Response {
    let path = request.uri().path();
    let host = request
        .headers()
        .get("host")
        .and_then(|h| h.to_str().ok())
        .unwrap_or("");
    if let Some(origin) = request
        .headers()
        .get("origin")
        .and_then(|h| h.to_str().ok())
    {
        if origin != format!("http://{host}") {
            return error(StatusCode::FORBIDDEN, "Cross-origin request rejected").into_response();
        }
    }
    if (path.starts_with("/v1/") && !path.starts_with("/v1/stream/"))
        || path == "/ready"
        || path == "/metrics"
    {
        let token = request
            .headers()
            .get("authorization")
            .and_then(|h| h.to_str().ok())
            .unwrap_or("")
            .strip_prefix("Bearer ")
            .unwrap_or("");
        if !authorized(&app.token, token) {
            return error(StatusCode::UNAUTHORIZED, "Bearer token required").into_response();
        }
    }
    let mut response = next.run(request).await;
    for (name, value) in [
        ("cache-control", "no-store"),
        ("x-content-type-options", "nosniff"),
        ("referrer-policy", "no-referrer"),
        (
            "content-security-policy",
            "default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; media-src 'self' blob:; img-src 'self' data:; object-src 'none'; frame-ancestors 'none'",
        ),
    ] {
        response
            .headers_mut()
            .insert(name, HeaderValue::from_static(value));
    }
    response
}
async fn metrics(State(app): State<Arc<App>>) -> Json<Value> {
    let sessions = app.sessions.lock().await;
    Json(
        json!({"runtime":"rust-python","active_sessions":sessions.len(),"sessions":sessions.values().map(|s|json!({"call_id":s.id,"generation":s.generation.load(Ordering::SeqCst),"capture":s.capture.lock().unwrap().stats()})).collect::<Vec<_>>()}),
    )
}
fn samples(config: &Value, key: &str) -> usize {
    (config[key].as_f64().unwrap() * RATE as f64).round() as usize
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let arg = |key: &str, default: &str| {
        args.windows(2)
            .find(|x| x[0] == key)
            .map(|x| x[1].clone())
            .unwrap_or_else(|| default.into())
    };
    let root = arg("--root", ".");
    let python = arg("--python", "python3");
    let port: u16 = arg("--port", "8000").parse()?;
    let deadline = Duration::from_millis(arg("--worker-timeout-ms", "5000").parse()?);
    let startup = Duration::from_millis(arg("--startup-timeout-ms", "30000").parse()?);
    let config: Value = serde_json::from_str(&std::env::var("STRIVE_NATIVE_CONFIG")?)?;
    let window = samples(&config, "window_s");
    let hop = samples(&config, "stride_s");
    if window == 0
        || window > RATE * 10
        || hop == 0
        || hop > window
        || config["stream_queue_windows"].as_u64().unwrap_or(0) == 0
    {
        return Err("Invalid native window configuration".into());
    }
    let worker = Worker::spawn(&python, &root, startup)
        .await
        .map_err(std::io::Error::other)?;
    let mut public = worker.hello["config"].clone();
    public["runtime"] = json!("rust-python");
    public["capabilities"] = json!({"live_audio":true,"upload":false,"demo":false});
    let app = Arc::new(App {
        sessions: Mutex::new(HashMap::new()),
        permits: Arc::new(Semaphore::new(
            config["max_sessions"].as_u64().unwrap() as usize
        )),
        spare: Mutex::new(Some(worker)),
        ready: json!({"ready":true,"runtime":"rust-python","deepfake_detection_validated":false}),
        token: config["api_token"].as_str().unwrap().into(),
        config,
        public,
        python,
        root: root.clone(),
        deadline,
        startup,
    });
    let state = app.clone();
    let reaper = tokio::spawn(async move {
        loop {
            sleep(Duration::from_secs(1)).await;
            let old: Vec<_> = state
                .sessions
                .lock()
                .await
                .values()
                .filter(|s| {
                    s.touched.lock().unwrap().elapsed().as_secs()
                        > state.config["idle_timeout_s"].as_u64().unwrap()
                })
                .cloned()
                .collect();
            for session in old {
                remove(&state, &session).await;
            }
        }
    });
    let router = Router::new()
        .route_service("/", ServeFile::new(format!("{root}/web/landing.html")))
        .route_service("/dashboard", ServeFile::new(format!("{root}/web/index.html")))
        .nest_service("/assets", ServeDir::new(format!("{root}/web")))
        .route(
            "/health",
            get(|| async { Json(json!({"status":"ok","runtime":"rust-python"})) }),
        )
        .route(
            "/ready",
            get(|State(a): State<Arc<App>>| async move { Json(a.ready.clone()) }),
        )
        .route(
            "/v1/config",
            get(|State(a): State<Arc<App>>| async move { Json(a.public.clone()) }),
        )
        .route("/metrics", get(metrics))
        .route("/v1/calls", post(start))
        .route("/v1/calls/{id}", delete(end))
        .route("/v1/calls/{id}/context", patch(context))
        .route("/v1/calls/{id}/verify", post(verify))
        .route("/v1/calls/{id}/transaction", post(transaction))
        .route("/v1/calls/{id}/audit", get(audit))
        .route("/v1/stream/{id}", get(websocket))
        .layer(DefaultBodyLimit::max(90000))
        .layer(middleware::from_fn_with_state(app.clone(), access))
        .with_state(app.clone());
    let listener = tokio::net::TcpListener::bind(("127.0.0.1", port)).await?;
    println!(
        "STRIVE Rust audio: http://127.0.0.1:{}",
        listener.local_addr()?.port()
    );
    let shutdown_app = app.clone();
    let listener = listener.tap_io(|socket| {
        if let Err(error) = socket.set_nodelay(true) {
            eprintln!("Could not disable TCP coalescing: {error}");
        }
    });
    axum::serve(listener, router)
        .with_graceful_shutdown(async move {
            let mut terminate =
                tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
                    .expect("signal handler");
            tokio::select! { _=tokio::signal::ctrl_c()=>{}, _=terminate.recv()=>{} }
            let remaining: Vec<_> = shutdown_app
                .sessions
                .lock()
                .await
                .values()
                .cloned()
                .collect();
            for session in remaining {
                remove(&shutdown_app, &session).await;
            }
        })
        .await?;
    reaper.abort();
    let remaining: Vec<_> = app.sessions.lock().await.values().cloned().collect();
    for session in remaining {
        remove(&app, &session).await;
    }
    Ok(())
}
