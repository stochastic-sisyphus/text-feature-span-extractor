import { useState, useEffect, useMemo } from 'react';
import { fetchConfig } from './api';
import { type FieldDetail } from './types';

export interface DisplayKeys {
  vendorKey: string | null;
  dateKey: string | null;
  idKey: string | null;
}

/**
 * Derives the three display keys (vendor, date, invoice ID) from a schema field list.
 * Selection is priority-ordered: required fields beat optional; falls back to the
 * highest-importance required field if a category has no match.
 */
export function useDisplayKeys(schemaFields: FieldDetail[]): DisplayKeys {
  return useMemo(() => {
    const sorted = [...schemaFields].sort((a, b) => b.importance - a.importance);
    const requiredSorted = sorted.filter((f) => f.required);
    const fallback = requiredSorted[0]?.name ?? null;

    const vendorKey =
      sorted.find((f) => f.normalizer === 'name' && f.required)?.name ??
      sorted.find((f) => f.normalizer === 'name')?.name ??
      fallback;

    const dateKey =
      sorted.find((f) => f.base_type === 'date' && f.required)?.name ??
      sorted.find((f) => f.base_type === 'date')?.name ??
      fallback;

    const idKey =
      sorted.find((f) => f.normalizer === 'id' && f.required)?.name ??
      sorted.find((f) => f.normalizer === 'id')?.name ??
      fallback;

    return { vendorKey, dateKey, idKey };
  }, [schemaFields]);
}

/**
 * Hook to fetch and cache confidence thresholds from config
 */
export function useConfidenceThresholds() {
  const [autoApproveThreshold, setAutoApproveThreshold] = useState(0.85);
  const [mediumThreshold, setMediumThreshold] = useState(0.5);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchConfig()
      .then(data => {
        if (data.confidence_auto_approve) {
          setAutoApproveThreshold(data.confidence_auto_approve);
        }
        if (data.confidence_thresholds?.medium) {
          setMediumThreshold(data.confidence_thresholds.medium);
        }
      })
      .catch(() => {
        // Fallback to defaults
      })
      .finally(() => {
        setLoading(false);
      });
  }, []);

  return { autoApproveThreshold, mediumThreshold, loading };
}
