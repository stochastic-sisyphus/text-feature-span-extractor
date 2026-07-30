import { buildQueueParams } from '../QueuePage';

// Case 1: filter='pending', sort='priority' → both present, no cursor
test('filter=pending, sort=priority sets mode=all_processed, no cursor', () => {
  const p = buildQueueParams({ sort: 'priority', filter: 'pending' });
  expect(p.get('filter')).toBe('pending');
  expect(p.get('sort')).toBe('priority');
  expect(p.has('cursor')).toBe(false);
  expect(p.get('mode')).toBe('all_processed');
});

// Case 2: with cursor → cursor param present
test('cursor is included when provided', () => {
  const p = buildQueueParams({ sort: 'recent', filter: 'ready', cursor: 'abc123' });
  expect(p.get('cursor')).toBe('abc123');
});

// Case 3: without cursor (undefined) → cursor absent
test('cursor is absent when undefined', () => {
  const p = buildQueueParams({ sort: 'recent', filter: 'ready', cursor: undefined });
  expect(p.has('cursor')).toBe(false);
});

// Case 4: mode is derived via filterToMode
// 'ready' → 'needs_review', 'all' → 'all', 'pending'/'failed' → 'all_processed'
test('filterToMode maps each filter correctly', () => {
  expect(buildQueueParams({ sort: 'recent', filter: 'ready' }).get('mode')).toBe('needs_review');
  expect(buildQueueParams({ sort: 'recent', filter: 'all' }).get('mode')).toBe('all');
  expect(buildQueueParams({ sort: 'recent', filter: 'failed' }).get('mode')).toBe('all_processed');
});
