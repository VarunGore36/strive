"use strict";

const stage = document.getElementById("signal-stage");
const canvas = document.getElementById("signal-canvas");
const header = document.getElementById("site-header");
const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
const context = canvas?.getContext("2d");

function setHeaderState() {
  header?.classList.toggle("scrolled", window.scrollY > 18);
}

window.addEventListener("scroll", setHeaderState, { passive: true });
setHeaderState();

if (canvas && context && stage) {
  let width = 0;
  let height = 0;
  let animationFrame = 0;
  let start = performance.now();
  const streams = [
    { y: .28, color: [233, 180, 76], phase: .2, amplitude: 14, side: "left" },
    { y: .38, color: [156, 166, 255], phase: 1.7, amplitude: 10, side: "right" },
    { y: .70, color: [83, 217, 156], phase: 3.4, amplitude: 8, side: "left" },
  ];

  function resize() {
    const ratio = Math.min(window.devicePixelRatio || 1, 2);
    width = stage.clientWidth;
    height = stage.clientHeight;
    canvas.width = Math.round(width * ratio);
    canvas.height = Math.round(height * ratio);
    context.setTransform(ratio, 0, 0, ratio, 0, 0);
  }

  function pointOnStream(stream, time, progress) {
    const centerX = width * .5;
    const centerY = height * .5;
    const startX = stream.side === "right" ? width * .92 : width * .08;
    const startY = height * stream.y;
    const direction = startX < centerX ? 1 : -1;
    const length = Math.max(1, Math.abs(centerX - startX) - 78);
    const x = startX + direction * length * progress;
    const baseline = startY + (centerY - startY) * Math.pow(progress, 1.8);
    const wave = Math.sin(progress * (8 + 7 * progress) * Math.PI + time + stream.phase);
    const y = baseline + wave * stream.amplitude * Math.sin(Math.PI * progress);
    return { x, y, startX, startY, centerX, centerY, length, direction };
  }

  function drawStream(stream, time) {
    const steps = 70;
    context.beginPath();
    for (let index = 0; index <= steps; index += 1) {
      const progress = index / steps;
      const point = pointOnStream(stream, time, progress);
      if (index === 0) context.moveTo(point.x, point.y);
      else context.lineTo(point.x, point.y);
    }
    const first = pointOnStream(stream, time, 0);
    const [red, green, blue] = stream.color;
    const gradient = context.createLinearGradient(first.startX, first.startY, first.centerX, first.centerY);
    gradient.addColorStop(0, `rgba(${red},${green},${blue},0)`);
    gradient.addColorStop(.2, `rgba(${red},${green},${blue},.5)`);
    gradient.addColorStop(1, `rgba(${red},${green},${blue},.12)`);
    context.strokeStyle = gradient;
    context.lineWidth = 1.4;
    context.stroke();

    for (let index = 0; index < 3; index += 1) {
      const progress = (time * .08 + index / 3 + stream.phase * .07) % 1;
      const point = pointOnStream(stream, time, progress);
      context.beginPath();
      context.arc(point.x, point.y, 2.1, 0, Math.PI * 2);
      context.fillStyle = `rgba(${red},${green},${blue},${.25 + progress * .7})`;
      context.fill();
    }
  }

  function paint(now) {
    const elapsed = reducedMotion.matches ? 2.5 : (now - start) / 1000;
    context.clearRect(0, 0, width, height);
    streams.forEach((stream) => drawStream(stream, elapsed));
    context.beginPath();
    context.arc(width * .5, height * .5, 108 + Math.sin(elapsed * 1.4) * 4, 0, Math.PI * 2);
    context.strokeStyle = "rgba(233,180,76,.10)";
    context.lineWidth = 1;
    context.stroke();
    if (!reducedMotion.matches) animationFrame = requestAnimationFrame(paint);
  }

  const resizeObserver = new ResizeObserver(resize);
  resizeObserver.observe(stage);
  resize();
  paint(performance.now());

  reducedMotion.addEventListener("change", () => {
    cancelAnimationFrame(animationFrame);
    start = performance.now();
    paint(performance.now());
  });
}
