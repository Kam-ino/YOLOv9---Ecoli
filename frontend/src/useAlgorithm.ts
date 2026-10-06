import { useCallback, useState } from 'react'
import { ALGORITHMS, type Algorithm } from './api'

const KEY = 'ecoli.algorithm'

// The user's detector choice, shared across tabs via localStorage so the
// Upload / Live / Label / Train pages agree without a global store.
export function useAlgorithm(): [Algorithm, (a: Algorithm) => void] {
  const [algo, setAlgo] = useState<Algorithm>(() => {
    try {
      const saved = localStorage.getItem(KEY)
      if (saved && (ALGORITHMS as string[]).includes(saved)) return saved as Algorithm
    } catch { /* private mode / blocked storage */ }
    return 'yolov9'
  })
  const set = useCallback((a: Algorithm) => {
    setAlgo(a)
    try { localStorage.setItem(KEY, a) } catch { /* ignore */ }
  }, [])
  return [algo, set]
}
