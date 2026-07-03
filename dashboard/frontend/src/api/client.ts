import axios from "axios";

const api = axios.create({ baseURL: "/api" });

export interface ModelEntry {
  key:           string;
  model:         string;
  train_dataset: string;
}

export interface EnsembleMember {
  key:        string;
  threshold?: number;
}

export interface ModelsResponse {
  gpu:             ModelEntry[];
  tflite:          ModelEntry[];
  gpu_ensemble:    EnsembleMember[];
  tflite_ensemble: EnsembleMember[];
  mean_threshold?: number;
}

export interface ConfigResponse {
  mode:      "gpu" | "tflite";
  model_key: string | null;
}

export interface LiveStatus {
  pipeline_loaded:   boolean;
  pipeline_error:    string | null;
  det_loaded:        boolean;
  available_gpu:     number;
  available_tflite:  number;
}

export interface EnsembleDetail {
  key:      string;
  mal_prob: number;
}

export interface InferenceResult {
  predicted_label:      "MALIGNANT" | "BENIGN" | null;
  malignant_prob:       number | null;
  benign_prob:          number | null;
  confidence:           number | null;
  bbox_xyxy:            [number, number, number, number] | null;
  detection_source:     string | null;
  detection_confidence: number | null;
  quality_score:        number | null;
  quality_ok:           boolean | null;
  crop_b64:             string | null;
  total_latency_ms:     number | null;
  stage_latencies_ms:   Record<string, number>;
  mode:                 string | null;
  model_key:            string | null;
  ensemble_details:     EnsembleDetail[];
  error:                string | null;
}

export interface HardwareSample {
  ts:           string;
  cpu_pct:      number;
  mem_pct:      number;
  cpu_temp_c:   number | null;
  gpu_temp_c:   number | null;
  gpu_power_w:  number | null;
  gpu_mem_mb:   number | null;
  gpu_util_pct: number | null;
}

export const getModels = () =>
  api.get<ModelsResponse>("/models").then((r) => r.data);

export const getConfig = () =>
  api.get<ConfigResponse>("/config").then((r) => r.data);

export const setConfig = (mode: string) =>
  api.post<ConfigResponse>("/config", { mode }).then((r) => r.data);

export const getLiveStatus = () =>
  api.get<LiveStatus>("/live/status").then((r) => r.data);

export const inferFrame = (base64Image: string, mode: string) =>
  api
    .post<InferenceResult>("/live/infer-frame", { image: base64Image, mode })
    .then((r) => r.data);

export const getHardwareCurrent = () =>
  api.get<HardwareSample>("/live/hardware/current").then((r) => r.data);

export const getHardwareHistory = (seconds = 120) =>
  api
    .get<{ samples: HardwareSample[] }>(`/live/hardware/history?seconds=${seconds}`)
    .then((r) => r.data.samples);
