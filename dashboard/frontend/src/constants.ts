export const CLASS_NAMES = [
  "mel", "nv", "bcc", "akiec", "bkl", "df", "vasc",
] as const;

export type ClassName = (typeof CLASS_NAMES)[number];

export const CLASS_DISPLAY: Record<string, string> = {
  mel:   "Melanoma",
  nv:    "Melanocytic Nevus",
  bcc:   "Basal Cell Carcinoma",
  akiec: "Actinic Keratosis",
  bkl:   "Benign Keratosis",
  df:    "Dermatofibroma",
  vasc:  "Vascular Lesion",
};

export const CLASS_COLORS: Record<string, string> = {
  mel:   "#EF4444",
  nv:    "#22C55E",
  bcc:   "#F97316",
  akiec: "#F59E0B",
  bkl:   "#3B82F6",
  df:    "#14B8A6",
  vasc:  "#A855F7",
};

// Classes that warrant clinical attention
export const HIGH_RISK: Set<string> = new Set(["mel", "bcc", "akiec"]);
