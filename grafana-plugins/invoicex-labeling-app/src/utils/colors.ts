// Semantic color constants
export const COLORS = {
  // Confidence/priority levels
  success: '#73BF69',
  warning: '#FF9830',
  error: '#F2495C',

  // Token selection overlays
  tokenSelected: 'rgba(34, 197, 94, 0.35)',
  tokenSelectedBorder: 'rgba(22, 163, 74, 0.9)',
  tokenHover: 'rgba(59, 130, 246, 0.15)',
  tokenHoverBorder: 'rgba(59, 130, 246, 0.7)',
} as const;

export const getStatusColor = (status: string) => {
  switch (status) {
    case 'PREDICTED': return 'green';
    case 'ABSTAIN': return 'orange';
    case 'MISSING': return 'red';
    default: return 'blue';
  }
};

export const getPriorityColor = (level: string) => {
  switch (level) {
    case 'urgent': return 'red';
    case 'medium': return 'orange';
    case 'low': return 'green';
    default: return 'blue';
  }
};

export const getConfidenceColor = (confidence: number, highThreshold?: number, mediumThreshold?: number) => {
  const high = highThreshold ?? 0.85;
  const medium = mediumThreshold ?? 0.5;
  if (confidence >= high) {return COLORS.success;}
  if (confidence >= medium) {return COLORS.warning;}
  return COLORS.error;
};

export const formatConfidencePercent = (confidence: number) => {
  return Math.round(confidence * 100);
};
