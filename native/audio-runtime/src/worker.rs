use serde_json::{Value, json};
use std::{
    os::{fd::OwnedFd, unix::net::UnixStream as StdStream},
    process::Stdio,
    time::Duration,
};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::UnixStream,
    process::{Child, Command},
    time::timeout,
};

pub struct Worker {
    child: Child,
    stream: UnixStream,
    pub hello: Value,
}

impl Worker {
    pub async fn spawn(python: &str, root: &str, startup: Duration) -> Result<Self, String> {
        let (parent, child_socket) = StdStream::pair().map_err(|e| e.to_string())?;
        parent.set_nonblocking(true).map_err(|e| e.to_string())?;
        let fd: OwnedFd = child_socket.into();
        let child = Command::new(python)
            .args(["-E", "-m", "strive.native_worker"])
            .current_dir(root)
            .stdin(Stdio::from(fd))
            .stdout(Stdio::null())
            .stderr(Stdio::inherit())
            .env("OMP_NUM_THREADS", "1")
            .env("OPENBLAS_NUM_THREADS", "1")
            .env("MKL_NUM_THREADS", "1")
            .kill_on_drop(true)
            .spawn()
            .map_err(|e| e.to_string())?;
        let mut worker = Self {
            child,
            stream: UnixStream::from_std(parent).map_err(|e| e.to_string())?,
            hello: Value::Null,
        };
        worker.hello = timeout(startup, worker.read())
            .await
            .map_err(|_| "Worker startup timed out")??;
        if worker.hello["ready"]["ready"] != true {
            return Err("Worker initialization failed".into());
        }
        Ok(worker)
    }

    async fn read(&mut self) -> Result<Value, String> {
        let size = self.stream.read_u32_le().await.map_err(|e| e.to_string())? as usize;
        if size == 0 || size > 1_000_000 {
            return Err("Invalid worker reply length".into());
        }
        let mut data = vec![0; size];
        self.stream
            .read_exact(&mut data)
            .await
            .map_err(|e| e.to_string())?;
        serde_json::from_slice(&data).map_err(|e| e.to_string())
    }

    pub async fn rpc(
        &mut self,
        header: &Value,
        samples: &[f32],
        deadline: Duration,
    ) -> Result<Value, String> {
        timeout(deadline, async {
            let encoded = serde_json::to_vec(header).map_err(|e| e.to_string())?;
            self.stream
                .write_u32_le(encoded.len() as u32)
                .await
                .map_err(|e| e.to_string())?;
            self.stream
                .write_u32_le((samples.len() * 4) as u32)
                .await
                .map_err(|e| e.to_string())?;
            self.stream
                .write_all(&encoded)
                .await
                .map_err(|e| e.to_string())?;
            let pcm: Vec<u8> = samples.iter().flat_map(|x| x.to_le_bytes()).collect();
            self.stream
                .write_all(&pcm)
                .await
                .map_err(|e| e.to_string())?;
            let response = self.read().await?;
            if response["request_id"] != header["request_id"]
                || response["generation"] != header["generation"]
            {
                return Err("Stale or mismatched worker reply".into());
            }
            if header["op"] == "window" && response["end"] != header["end"] {
                return Err("Wrong window in worker reply".into());
            }
            if response.get("error").is_some() {
                return Err(format!("Worker error: {}", response["error"]));
            }
            Ok(response)
        })
        .await
        .map_err(|_| "Worker deadline exceeded".to_owned())?
    }

    pub async fn init(
        &mut self,
        body: Value,
        generation: u64,
        deadline: Duration,
    ) -> Result<Value, String> {
        self.rpc(
            &json!({"op":"init","body":body,"generation":generation,"request_id":0}),
            &[],
            deadline,
        )
        .await
    }

    pub async fn stop(&mut self) {
        let _ = self.child.start_kill();
        let _ = timeout(Duration::from_secs(2), self.child.wait()).await;
    }
}
