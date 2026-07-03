import {
  useCallback,
  useEffect,
  useRef,
  useState,
} from "react";
import {
  getLiveStatus, inferFrame,
  type InferenceResult, type LiveStatus,
  type EnsembleDetail,
} from "../api/client";
import { theme } from "../theme";

const INFER_INTERVAL_MS  = 50;
const RESULT_UPDATE_MS   = 1000;   // diagnosis panel refresh rate
const CAPTURE_W = 640;
const CAPTURE_H = 480;
const OVERLAY_FPS      = 24;
const OVERLAY_FRAME_MS = 1000 / OVERLAY_FPS;
const LERP_ALPHA = 0.12;

const MOTION_W         = 80;
const MOTION_H         = 60;
const MOTION_THRESHOLD = 15;
const STABLE_FRAMES    = 3;

const COLOR_MALIGNANT = "#EF4444";
const COLOR_BENIGN    = "#22C55E";

function fmtKey(key: string): string {
  const [model, ds] = key.split("|");
  const modelLabel  = model
    .replace("efficientnet_b2",  "EfficientNet-B2")
    .replace("convnext_tiny_se", "ConvNeXt-Tiny-SE")
    .replace("medfusionnet",     "MedFusionNet")
    .replace("resnet50",         "ResNet-50")
    .replace("mobilenetv3_large","MobileNetV3")
    .replace("yolov8_cls",       "YOLOv8-cls")
    .replace("meta_learner",     "Meta-Learner");
  return `${modelLabel} / ${(ds ?? "").toUpperCase()}`;
}

function Label({ children, style }: { children: React.ReactNode; style?: React.CSSProperties }) {
  return (
    <span style={{
      fontSize: 10, fontWeight: 700, letterSpacing: "0.1em",
      textTransform: "uppercase", color: theme.textMuted, ...style,
    }}>
      {children}
    </span>
  );
}

function Card({ children, style, glow }: {
  children: React.ReactNode;
  style?: React.CSSProperties;
  glow?: string;
}) {
  return (
    <div style={{
      background: theme.bgCard,
      border: `1px solid ${glow ? glow + "55" : theme.border}`,
      borderRadius: 12, padding: "16px",
      boxShadow: glow ? `0 0 20px ${glow}22` : undefined,
      ...style,
    }}>
      {children}
    </div>
  );
}

function EnsembleModels({ details }: { details: EnsembleDetail[] }) {
  if (details.length === 0) return null;
  return (
    <div style={{ marginTop: 14, borderTop: `1px solid ${theme.border}`, paddingTop: 12 }}>
      <Label style={{ display: "block", marginBottom: 8 }}>Model Probabilities</Label>
      <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
        {details.map((d) => {
          const malPct = d.mal_prob * 100;
          const color  = malPct >= 50 ? COLOR_MALIGNANT : COLOR_BENIGN;
          return (
            <div key={d.key}>
              <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 3 }}>
                <span style={{ fontSize: 11, color: theme.textPrimary }}>{fmtKey(d.key)}</span>
                <span style={{ fontSize: 11, color, fontWeight: 700 }}>{malPct.toFixed(1)}%</span>
              </div>
              <div style={{ height: 6, background: theme.bgInput, borderRadius: 3, overflow: "hidden" }}>
                <div style={{
                  width: `${malPct}%`, height: "100%",
                  background: color, borderRadius: 3, transition: "width 0.3s ease",
                }} />
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function DiagnosisBadge({ result }: { result: InferenceResult | null }) {
  if (!result) {
    return (
      <div style={{ textAlign: "center", padding: "24px 0", color: theme.textMuted, fontSize: 14 }}>
        Awaiting frame…
      </div>
    );
  }

  if (result.quality_ok === false) {
    return (
      <div style={{
        background: theme.warning + "22", border: `2px solid ${theme.warning}`,
        borderRadius: 12, padding: "16px 20px", textAlign: "center",
      }}>
        <div style={{ fontSize: 14, fontWeight: 800, color: theme.warning }}>LOW IMAGE QUALITY</div>
        <div style={{ fontSize: 12, color: theme.textMuted, marginTop: 4 }}>
          Blur score: {result.quality_score?.toFixed(1) ?? "—"} (min 80 required)
        </div>
        <div style={{ fontSize: 11, color: theme.textMuted, marginTop: 4 }}>
          Move closer or hold camera steady
        </div>
      </div>
    );
  }

  if (!result.predicted_label) {
    return (
      <div style={{ textAlign: "center", padding: "24px 0", color: theme.textMuted, fontSize: 14 }}>
        {result.error
          ? <span style={{ color: theme.error }}>{result.error}</span>
          : "Detecting…"}
      </div>
    );
  }

  const isMal = result.predicted_label === "MALIGNANT";
  const color = isMal ? COLOR_MALIGNANT : COLOR_BENIGN;
  const malP  = result.malignant_prob ?? 0;
  const benP  = result.benign_prob ?? 0;

  return (
    <div>
      <div style={{
        background: color + "22", border: `2px solid ${color}`,
        borderRadius: 12, padding: "16px 20px", textAlign: "center",
        boxShadow: `0 0 24px ${color}33`,
      }}>
        {isMal && (
          <div style={{ fontSize: 10, fontWeight: 800, letterSpacing: "0.15em", color, marginBottom: 4 }}>
            CLINICAL REVIEW RECOMMENDED
          </div>
        )}
        <div style={{ fontSize: 26, fontWeight: 900, color, lineHeight: 1.2 }}>
          {result.predicted_label}
        </div>
        <div style={{ marginTop: 6, fontSize: 32, fontWeight: 900, color: theme.textPrimary }}>
          {((result.confidence ?? 0) * 100).toFixed(1)}%
        </div>
        <div style={{ fontSize: 11, color: theme.textMuted, marginTop: 2 }}>confidence</div>
      </div>

      <div style={{ marginTop: 12, display: "flex", flexDirection: "column", gap: 6 }}>
        {[
          { label: "Malignant", prob: malP, color: COLOR_MALIGNANT },
          { label: "Benign",    prob: benP, color: COLOR_BENIGN },
        ].map(({ label, prob, color: c }) => (
          <div key={label} style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <span style={{ width: 80, fontSize: 12, color: c, fontWeight: 700 }}>{label}</span>
            <div style={{
              flex: 1, height: 10, background: theme.bgInput,
              borderRadius: 5, overflow: "hidden",
            }}>
              <div style={{
                width: `${prob * 100}%`, height: "100%",
                background: c, borderRadius: 5, transition: "width 0.3s ease",
              }} />
            </div>
            <span style={{ width: 44, textAlign: "right", fontSize: 12, color: c, fontWeight: 700 }}>
              {(prob * 100).toFixed(1)}%
            </span>
          </div>
        ))}
      </div>

      <EnsembleModels details={result.ensemble_details ?? []} />
    </div>
  );
}

function CropThumbnail({ cropB64, result }: {
  cropB64: string | null;
  result:  InferenceResult | null;
}) {
  if (!cropB64) {
    return (
      <div style={{
        aspectRatio: "1/1", background: theme.bgInput, borderRadius: 8,
        display: "flex", alignItems: "center", justifyContent: "center",
      }}>
        <span style={{ fontSize: 12, color: theme.textMuted }}>No crop yet</span>
      </div>
    );
  }

  const isMal  = result?.predicted_label === "MALIGNANT";
  const border = result?.predicted_label
    ? `2px solid ${isMal ? COLOR_MALIGNANT : COLOR_BENIGN}`
    : `1px solid ${theme.border}`;

  return (
    <div style={{ position: "relative" }}>
      <img
        src={cropB64}
        alt="Lesion crop"
        style={{
          width: "100%", aspectRatio: "1/1", objectFit: "contain",
          borderRadius: 8, border, display: "block",
          background: "#000",
          transition: "opacity 0.25s ease",
        }}
      />
      {result?.quality_ok === false && (
        <div style={{
          position: "absolute", inset: 0, borderRadius: 8,
          background: "rgba(0,0,0,0.55)",
          display: "flex", alignItems: "center", justifyContent: "center",
        }}>
          <span style={{ color: theme.warning, fontSize: 12, fontWeight: 700 }}>BLURRY</span>
        </div>
      )}
    </div>
  );
}


function _iou(
  a: [number, number, number, number],
  b: [number, number, number, number],
): number {
  const ix1 = Math.max(a[0], b[0]), iy1 = Math.max(a[1], b[1]);
  const ix2 = Math.min(a[2], b[2]), iy2 = Math.min(a[3], b[3]);
  const inter = Math.max(0, ix2 - ix1) * Math.max(0, iy2 - iy1);
  if (inter === 0) return 0;
  const areaA = (a[2] - a[0]) * (a[3] - a[1]);
  const areaB = (b[2] - b[0]) * (b[3] - b[1]);
  return inter / (areaA + areaB - inter);
}

export function LivePage() {
  const videoRef         = useRef<HTMLVideoElement>(null);
  const overlayCanvasRef = useRef<HTMLCanvasElement>(null);
  const captureCanvasRef = useRef<HTMLCanvasElement>(null);
  const motionCanvasRef  = useRef<HTMLCanvasElement>(null);

  const [cameraError, setCameraError] = useState<string | null>(null);
  const [cameraReady, setCameraReady] = useState(false);
  const [capturing,   setCapturing]   = useState(false);

  const [status,  setStatus]  = useState<LiveStatus | null>(null);
  const [result,  setResult]  = useState<InferenceResult | null>(null);
  const [cropB64, setCropB64] = useState<string | null>(null);
  const lastCropTsRef    = useRef(0);
  const lastResultTsRef  = useRef(0);
  const pendingBboxRef   = useRef<[number, number, number, number] | null>(null);
  const pendingHitsRef   = useRef(0);

  const modeRef = useRef<"gpu">("gpu");
  const inferringRef  = useRef(false);
  const targetBboxRef  = useRef<[number, number, number, number] | null>(null);
  const displayBboxRef = useRef<[number, number, number, number] | null>(null);
  const overlayResRef  = useRef<InferenceResult | null>(null);

  const prevFrameRef    = useRef<Uint8ClampedArray | null>(null);
  const stableCountRef  = useRef(0);
  const cameraStableRef = useRef(false);
  const [cameraStable, setCameraStable] = useState(false);


  const startCamera = useCallback(async () => {
    setCameraError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: CAPTURE_W }, height: { ideal: CAPTURE_H }, facingMode: "environment" },
        audio: false,
      });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.onloadedmetadata = () => {
          videoRef.current?.play();
          setCameraReady(true);
        };
      }
    } catch (err) {
      setCameraError(
        err instanceof DOMException && err.name === "NotFoundError"
          ? "No camera found on this device."
          : `Camera error: ${(err as Error).message}`
      );
    }
  }, []);

  const updateOverlay = useCallback((res: InferenceResult | null) => {
    overlayResRef.current = res;

    if (!res?.bbox_xyxy || res.detection_source !== "yolov8") {
      pendingBboxRef.current = null;
      pendingHitsRef.current = 0;
      if (res?.detection_source === "centre_crop") targetBboxRef.current = null;
      return;
    }

    const next = res.bbox_xyxy;
    const prev = pendingBboxRef.current;

    if (prev && _iou(prev, next) >= 0.4) {
      pendingHitsRef.current += 1;
    } else {
      pendingBboxRef.current = next;
      pendingHitsRef.current = 1;
      return;
    }

    if (pendingHitsRef.current >= 2) {
      targetBboxRef.current  = next;
      pendingBboxRef.current = next;
    }
  }, []);

  // smooth lerp + canvas redraw, capped at 24fps
  useEffect(() => {
    if (!cameraReady) return;
    let rafId: number;
    let lastTs = 0;

    const frame = (ts: number) => {
      rafId = requestAnimationFrame(frame);
      if (ts - lastTs < OVERLAY_FRAME_MS) return;
      lastTs = ts;

      const canvas = overlayCanvasRef.current;
      const video  = videoRef.current;
      if (canvas && video) {
        const ctx = canvas.getContext("2d");
        if (ctx) {
          // canvas is always pinned to capture dimensions — bbox coords map 1:1
          if (canvas.width  !== CAPTURE_W) canvas.width  = CAPTURE_W;
          if (canvas.height !== CAPTURE_H) canvas.height = CAPTURE_H;
          ctx.clearRect(0, 0, CAPTURE_W, CAPTURE_H);

          const target = targetBboxRef.current;
          if (target) {
            if (!displayBboxRef.current) {
              displayBboxRef.current = [...target] as [number, number, number, number];
            } else {
              displayBboxRef.current = displayBboxRef.current.map(
                (v, i) => v + (target[i] - v) * LERP_ALPHA
              ) as [number, number, number, number];
            }

            const res = overlayResRef.current;
            const [sx1, sy1, sx2, sy2] = displayBboxRef.current;

            const boxColor = res?.predicted_label === "MALIGNANT" ? COLOR_MALIGNANT : COLOR_BENIGN;

            ctx.shadowColor = boxColor;
            ctx.shadowBlur  = 14;
            ctx.strokeStyle = boxColor;
            ctx.lineWidth   = 2;
            ctx.setLineDash([]);
            ctx.strokeRect(sx1, sy1, sx2 - sx1, sy2 - sy1);

            ctx.shadowBlur = 0;
            ctx.lineWidth  = 3;
            const cLen = 18;
            ([
              [sx1, sy1, sx1 + cLen, sy1], [sx1, sy1, sx1, sy1 + cLen],
              [sx2, sy1, sx2 - cLen, sy1], [sx2, sy1, sx2, sy1 + cLen],
              [sx1, sy2, sx1 + cLen, sy2], [sx1, sy2, sx1, sy2 - cLen],
              [sx2, sy2, sx2 - cLen, sy2], [sx2, sy2, sx2, sy2 - cLen],
            ] as number[][]).forEach(([ax, ay, bx, by]) => {
              ctx.beginPath(); ctx.moveTo(ax, ay); ctx.lineTo(bx, by); ctx.stroke();
            });

            if (res?.quality_ok === false) {
              ctx.fillStyle = "rgba(0,0,0,0.7)";
              ctx.fillRect(sx1, sy2 + 4, 110, 22);
              ctx.fillStyle = theme.warning;
              ctx.font = "bold 11px -apple-system, sans-serif";
              ctx.fillText(`BLURRY (${res.quality_score?.toFixed(0)})`, sx1 + 6, sy2 + 19);
            }

            if (res?.predicted_label) {
              const confStr = res.confidence != null ? ` ${(res.confidence * 100).toFixed(0)}%` : "";
              const text    = `${res.predicted_label}${confStr}`;
              const pad     = 6;
              ctx.font      = "bold 13px -apple-system, sans-serif";
              const tw      = ctx.measureText(text).width;
              const bx2     = sx1;
              const by2     = Math.max(0, sy1 - 30);
              ctx.fillStyle = boxColor + "DD";
              ctx.fillRect(bx2, by2, tw + pad * 2, 26);
              ctx.fillStyle = "#ffffff";
              ctx.fillText(text, bx2 + pad, by2 + 17);
            }
          }
        }
      }

      // motion stability check
      const mCanvas = motionCanvasRef.current;
      if (mCanvas && video) {
        const mCtx = mCanvas.getContext("2d", { willReadFrequently: true });
        if (mCtx) {
          mCtx.drawImage(video, 0, 0, MOTION_W, MOTION_H);
          const curr = mCtx.getImageData(0, 0, MOTION_W, MOTION_H).data;
          const prev = prevFrameRef.current;
          if (prev) {
            let totalDiff = 0;
            for (let i = 0; i < curr.length; i += 4) {
              totalDiff += Math.abs(curr[i]   - prev[i]);
              totalDiff += Math.abs(curr[i+1] - prev[i+1]);
              totalDiff += Math.abs(curr[i+2] - prev[i+2]);
            }
            const meanDiff = totalDiff / (MOTION_W * MOTION_H * 3);
            const moving = meanDiff >= MOTION_THRESHOLD;

            if (!moving) {
              stableCountRef.current = Math.min(stableCountRef.current + 1, STABLE_FRAMES);
            } else {
              stableCountRef.current = 0;
            }
            const nowStable = stableCountRef.current >= STABLE_FRAMES;
            if (nowStable !== cameraStableRef.current) {
              cameraStableRef.current = nowStable;
              setCameraStable(nowStable);
            }
          }
          prevFrameRef.current = curr.slice();
        }
      }
    };

    rafId = requestAnimationFrame(frame);
    return () => cancelAnimationFrame(rafId);
  }, [cameraReady]);

  const captureAndInfer = useCallback(async () => {
    const video  = videoRef.current;
    const canvas = captureCanvasRef.current;
    if (!video || !canvas || !cameraReady || inferringRef.current || !cameraStableRef.current) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    canvas.width  = CAPTURE_W;
    canvas.height = CAPTURE_H;
    ctx.drawImage(video, 0, 0, CAPTURE_W, CAPTURE_H);
    const base64 = canvas.toDataURL("image/jpeg", 0.8);

    inferringRef.current = true;
    try {
      const res = await inferFrame(base64, modeRef.current);
      updateOverlay(res);

      const now = Date.now();

      // throttle the diagnosis panel to RESULT_UPDATE_MS
      if (now - lastResultTsRef.current >= RESULT_UPDATE_MS) {
        setResult(res);
        lastResultTsRef.current = now;
      }

      // only update the crop when YOLO found a lesion, not on the centre-crop fallback
      if (res.crop_b64 && res.detection_source === "yolov8") {
        if (now - lastCropTsRef.current >= 800) {
          setCropB64(res.crop_b64);
          lastCropTsRef.current = now;
        }
      }
    } catch {
      /* keep last result on network error */
    } finally {
      inferringRef.current = false;
    }
  }, [cameraReady, updateOverlay]);

  useEffect(() => {
    startCamera();

    getLiveStatus().then(setStatus).catch(() => {});

    const statusInterval = setInterval(() => {
      getLiveStatus().then(setStatus).catch(() => {});
    }, 5000);

    return () => {
      clearInterval(statusInterval);
      if (videoRef.current?.srcObject) {
        (videoRef.current.srcObject as MediaStream).getTracks().forEach((t) => t.stop());
      }
    };
  }, [startCamera]);

  useEffect(() => {
    if (!capturing || !cameraReady) return;
    let cancelled = false;

    const loop = async () => {
      while (!cancelled) {
        if (!cameraStableRef.current) {
          await new Promise<void>((r) => setTimeout(r, 50));
          continue;
        }
        const t0 = Date.now();
        await captureAndInfer();
        if (!cancelled) {
          const wait = Math.max(0, INFER_INTERVAL_MS - (Date.now() - t0));
          if (wait > 0) await new Promise<void>((r) => setTimeout(r, wait));
        }
      }
    };

    loop();
    return () => { cancelled = true; };
  }, [capturing, cameraReady, captureAndInfer]);

  const pipelineReady = status != null &&
    (status.available_gpu > 0 || status.available_tflite > 0);

  return (
    <div style={{ minHeight: "100vh", background: theme.bg, display: "flex", flexDirection: "column" }}>

      {/* Header */}
      <header style={{
        display: "flex", alignItems: "center", justifyContent: "space-between",
        padding: "12px 24px",
        background: theme.bgCard, borderBottom: `1px solid ${theme.border}`,
        flexWrap: "wrap", gap: 10,
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          <div style={{
            width: 10, height: 10, borderRadius: "50%",
            background: pipelineReady ? theme.success : theme.warning,
            boxShadow: `0 0 8px ${pipelineReady ? theme.success : theme.warning}`,
          }} />
          <span style={{ fontSize: 15, fontWeight: 800, letterSpacing: "0.05em", color: theme.textPrimary }}>
            MELANOMA DETECTION
          </span>
          <span style={{ fontSize: 11, color: theme.textMuted, visibility: status ? "visible" : "hidden" }}>
            {status?.det_loaded ? "Det ✓" : "Det ✗"}
          </span>
        </div>


        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          {capturing && (
            <button
              onClick={() => {
                setCapturing(false);
                setResult(null);
                setCropB64(null);
                lastCropTsRef.current   = 0;
                lastResultTsRef.current = 0;
                pendingBboxRef.current = null;
                pendingHitsRef.current = 0;
                overlayResRef.current  = null;
                targetBboxRef.current  = null;
                displayBboxRef.current = null;
                prevFrameRef.current   = null;
                stableCountRef.current = 0;
                cameraStableRef.current = false;
                setCameraStable(false);
              }}
              style={{
                padding: "8px 20px", borderRadius: 8, border: "none",
                cursor: "pointer", fontWeight: 700, fontSize: 13,
                letterSpacing: "0.05em", background: theme.error, color: "#fff",
                transition: "background 0.2s",
              }}
            >
              STOP
            </button>
          )}
        </div>
      </header>

      {/* Main content */}
      <div style={{
        flex: 1, display: "grid",
        gridTemplateColumns: "1fr 360px",
        gap: 16, padding: 16,
        maxWidth: 1400, width: "100%", margin: "0 auto", alignItems: "start",
      }}>

        {/* Left: camera + hardware */}
        <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>

          {/* Camera feed */}
          <Card style={{ padding: 0, overflow: "hidden" }}>
            <div
              onClick={() => {
                if (cameraReady && !capturing) {
                  inferringRef.current    = false;
                  targetBboxRef.current   = null;
                  displayBboxRef.current  = null;
                  overlayResRef.current   = null;
                  prevFrameRef.current    = null;
                  stableCountRef.current  = 0;
                  cameraStableRef.current = false;
                  setCameraStable(false);
                  setResult(null);
                  setCapturing(true);
                }
              }}
              style={{
                position: "relative", width: "100%", background: "#000",
                aspectRatio: "4/3", borderRadius: 12, overflow: "hidden",
                cursor: cameraReady && !capturing ? "pointer" : "default",
              }}
            >
              <video
                ref={videoRef} muted playsInline
                style={{
                  width: "100%", height: "100%", objectFit: "fill",
                  display: cameraError ? "none" : "block",
                }}
              />
              <canvas
                ref={overlayCanvasRef}
                style={{
                  position: "absolute", inset: 0,
                  width: "100%", height: "100%", pointerEvents: "none",
                }}
              />

              {cameraError && (
                <div style={{
                  display: "flex", flexDirection: "column",
                  alignItems: "center", justifyContent: "center",
                  height: "100%", gap: 12, color: theme.textSecondary,
                }}>
                  <span style={{ fontSize: 48 }}>📷</span>
                  <span style={{ fontSize: 14 }}>{cameraError}</span>
                  <button onClick={startCamera} style={{
                    marginTop: 8, padding: "8px 20px", background: theme.blue,
                    color: "#fff", border: "none", borderRadius: 8,
                    cursor: "pointer", fontWeight: 700,
                  }}>
                    Retry Camera
                  </button>
                </div>
              )}

              {!capturing && !cameraError && cameraReady && (
                <div style={{
                  position: "absolute", inset: 0,
                  display: "flex", flexDirection: "column",
                  alignItems: "center", justifyContent: "center",
                  background: "rgba(0,0,0,0.32)", pointerEvents: "none",
                }}>
                  <div style={{
                    fontSize: 22, fontWeight: 800, color: "#fff",
                    letterSpacing: "0.07em", textAlign: "center",
                    animation: "clickToBlink 1.3s ease-in-out infinite",
                    textShadow: "0 2px 18px rgba(0,0,0,0.95)",
                  }}>
                    ▶ CLICK TO START
                  </div>
                  <div style={{
                    fontSize: 12, color: "rgba(255,255,255,0.5)",
                    marginTop: 10, letterSpacing: "0.05em",
                  }}>
                    tap anywhere on the feed to begin detection
                  </div>
                </div>
              )}

              {/* SEEKING / HOLD STEADY badge */}
              {capturing && (
                <div style={{
                  position: "absolute", top: 12, right: 12,
                  display: "flex", alignItems: "center", gap: 6,
                  background: "rgba(0,0,0,0.6)", padding: "4px 10px", borderRadius: 20,
                }}>
                  <span style={{
                    width: 8, height: 8, borderRadius: "50%",
                    background: cameraStable ? theme.error : theme.warning,
                    animation: cameraStable ? "pulse 1s infinite" : undefined,
                  }} />
                  <span style={{ fontSize: 11, color: "#fff", fontWeight: 700 }}>
                    {cameraStable ? "SEEKING" : "HOLD STEADY"}
                  </span>
                </div>
              )}

              {/* Pipeline stage circles */}
              {capturing && result && (
                <div style={{
                  position: "absolute", bottom: 12, left: 12,
                  display: "flex", gap: 4,
                }}>
                  {[
                    { key: "detect",   label: "D" },
                    { key: "quality",  label: "Q" },
                    { key: "classify", label: "C" },
                  ].map(({ key, label }) => {
                    const done = key in (result.stage_latencies_ms ?? {});
                    return (
                      <div key={key} style={{
                        width: 22, height: 22, borderRadius: "50%",
                        background: done ? COLOR_BENIGN + "CC" : "rgba(255,255,255,0.2)",
                        display: "flex", alignItems: "center", justifyContent: "center",
                        fontSize: 9, fontWeight: 800, color: "#fff",
                      }}>
                        {label}
                      </div>
                    );
                  })}
                </div>
              )}
            </div>
          </Card>

        </div>

        {/* Right panel */}
        <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>

          {/* Diagnosis */}
          <Card glow={result?.predicted_label === "MALIGNANT" ? COLOR_MALIGNANT
            : result?.predicted_label === "BENIGN" ? COLOR_BENIGN : undefined}>
            <Label style={{ display: "block", marginBottom: 10 }}>Diagnosis</Label>
            <DiagnosisBadge result={result} />
          </Card>

          {/* Lesion crop */}
          <Card>
            <Label style={{ display: "block", marginBottom: 10 }}>
              Lesion Crop
              {result?.quality_score != null && (
                <span style={{
                  marginLeft: 8, fontWeight: 400,
                  color: (result.quality_ok) ? COLOR_BENIGN : theme.warning,
                }}>
                  — blur score: {result.quality_score.toFixed(0)}
                </span>
              )}
            </Label>
            <CropThumbnail cropB64={cropB64} result={result} />
            {result?.detection_source === "yolov8" && (
              <div style={{ marginTop: 6, fontSize: 11, color: theme.textMuted, textAlign: "center" }}>
                Detection: yolov8
                {result.detection_confidence != null && result.detection_confidence > 0 &&
                  ` (${(result.detection_confidence * 100).toFixed(0)}%)`}
              </div>
            )}
          </Card>


{result?.error && (
            <Card style={{ borderColor: theme.error + "55", background: theme.error + "11" }}>
              <Label style={{ color: theme.error, display: "block", marginBottom: 6 }}>
                Inference Error
              </Label>
              <span style={{ fontSize: 12, color: theme.error }}>{result.error}</span>
            </Card>
          )}
        </div>
      </div>

      <style>{`
        @keyframes pulse        { 0%, 100% { opacity: 1; } 50% { opacity: 0.3;  } }
        @keyframes clickToBlink { 0%, 100% { opacity: 1; } 50% { opacity: 0.15; } }
      `}</style>
      <canvas ref={captureCanvasRef} style={{ display: "none" }} />
      <canvas ref={motionCanvasRef} width={MOTION_W} height={MOTION_H} style={{ display: "none" }} />
    </div>
  );
}
