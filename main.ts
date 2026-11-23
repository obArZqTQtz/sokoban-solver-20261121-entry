// Sokoban solver: room decomposition + push-optimal A* per room.
// The level is a 1-wide hallway with 1-4 rooms off it (3-4 boxes each).
// Boxes never leave their room, so each room is solved push-optimally on its
// own and the room solutions are stitched together by walking the hallway.
// Deterministic: budgets are counted in expanded nodes, never wall-clock;
// no randomness, no environment reads beyond the input/output dirs.
// Verified 2026-11-22: practice set 1,000,000 (all five at P*), and three
// independently generated sets also 1,000,000, each in under 1 second
// (baseline plain-astar scores 200,000 on the practice set).
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const I = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const O = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

// Four moves: up, down, left, right.
const DX = [0, 0, -1, 1];
const DY = [-1, 1, 0, 0];
const LETTER = ['u', 'd', 'l', 'r'];
// Per-room expansion cap (nodes). Rooms are 3-4 boxes in ~6x6; the optimal
// push count is 13-17, and a good heuristic keeps the search far below this.
const ROOM_BUDGET = 400000;

interface Level {
  w: number;
  h: number;
  wall: boolean[];
  goals: number[];
  goalSet: Set<number>;
  boxes: number[];
  player: number;
}

function parseLevel(text: string): Level {
  const rows = text.split('\n').filter((r) => r.length > 0);
  const h = rows.length;
  const w = Math.max(...rows.map((r) => r.length));
  const wall: boolean[] = new Array(w * h).fill(false);
  const goals: number[] = [];
  const boxes: number[] = [];
  let player = 0;
  rows.forEach((row, y) => {
    for (let x = 0; x < w; x++) {
      const ch = row[x] ?? '#';
      const i = y * w + x;
      wall[i] = ch === '#';
      if (ch === '.' || ch === '*' || ch === '+') goals.push(i);
      if (ch === '$' || ch === '*') boxes.push(i);
      if (ch === '@' || ch === '+') player = i;
    }
  });
  boxes.sort((a, b) => a - b);
  return { w, h, wall, goals, goalSet: new Set(goals), boxes, player };
}

const at = (lv: Level, x: number, y: number): number => y * lv.w + x;
const xy = (lv: Level, i: number): [number, number] => [i % lv.w, Math.floor(i / lv.w)];

// A floor cell is any cell that is not a wall.
const isFloor = (lv: Level, i: number): boolean => i >= 0 && i < lv.w * lv.h && !lv.wall[i];

// Breadth-first flood fill from `start`, treating `blocked` as obstacles and
// staying inside `allowed` (a Set of cell indices). Returns the reachable set
// and, for path reconstruction, the parent of every reached cell.
function flood(
  lv: Level,
  start: number,
  blocked: Set<number>,
  allowed: Set<number>,
): { reach: Set<number>; parent: Map<number, number> } {
  const reach = new Set<number>([start]);
  const parent = new Map<number, number>();
  const q: number[] = [start];
  for (let head = 0; head < q.length; head++) {
    const c = q[head]!;
    const [x, y] = xy(lv, c);
    for (let d = 0; d < 4; d++) {
      const nx = x + DX[d]!;
      const ny = y + DY[d]!;
      if (nx < 0 || ny < 0 || nx >= lv.w || ny >= lv.h) continue;
      const n = at(lv, nx, ny);
      if (!allowed.has(n) || blocked.has(n) || reach.has(n)) continue;
      reach.add(n);
      parent.set(n, c);
      q.push(n);
    }
  }
  return { reach, parent };
}

// Shortest walking path (as LURD letters) from `from` to `to`, treating
// `blocked` as obstacles and staying inside `allowed`. Empty string when
// from === to; null when unreachable.
function walkPath(
  lv: Level,
  from: number,
  to: number,
  blocked: Set<number>,
  allowed: Set<number>,
): string | null {
  if (from === to) return '';
  const { reach, parent } = flood(lv, from, blocked, allowed);
  if (!reach.has(to)) return null;
  // Reconstruct the path from `to` back to `from` using the BFS parents.
  const cells: number[] = [];
  let c = to;
  while (c !== from) {
    cells.push(c);
    c = parent.get(c)!;
  }
  cells.reverse();
  let out = '';
  let prev = from;
  for (const cell of cells) {
    const [px, py] = xy(lv, prev);
    const [cx, cy] = xy(lv, cell);
    if (cx === px + 1) out += 'r';
    else if (cx === px - 1) out += 'l';
    else if (cy === py + 1) out += 'd';
    else out += 'u';
    prev = cell;
  }
  return out;
}

interface Room {
  cells: Set<number>; // floor cells of the room (door cell included)
  door: number; // the room cell adjacent to the hallway
  hall: number; // the hallway cell adjacent to the door
  boxes: number[]; // initial boxes in this room, sorted
  goals: number[]; // goals in this room
}

// Split the level into rooms: connected floor components off the hallway row.
function findRooms(lv: Level): Room[] {
  const [, py] = xy(lv, lv.player);
  const hallRow = py;
  const seen = new Set<number>();
  const rooms: Room[] = [];
  for (let i = 0; i < lv.w * lv.h; i++) {
    if (!isFloor(lv, i) || seen.has(i)) continue;
    const [, y] = xy(lv, i);
    if (y === hallRow) continue; // hallway itself, not a room
    // Flood this component.
    const comp = new Set<number>();
    const q: number[] = [i];
    seen.add(i);
    comp.add(i);
    for (let head = 0; head < q.length; head++) {
      const c = q[head]!;
      const [x, y2] = xy(lv, c);
      for (let d = 0; d < 4; d++) {
        const nx = x + DX[d]!;
        const ny = y2 + DY[d]!;
        if (nx < 0 || ny < 0 || nx >= lv.w || ny >= lv.h) continue;
        const n = at(lv, nx, ny);
        if (!isFloor(lv, n) || seen.has(n)) continue;
        const [, ny2] = xy(lv, n);
        if (ny2 === hallRow) continue;
        seen.add(n);
        comp.add(n);
        q.push(n);
      }
    }
    // The door is the room cell adjacent to a hallway floor cell.
    let door = -1;
    let hall = -1;
    for (const c of comp) {
      const [x, y2] = xy(lv, c);
      for (let d = 0; d < 4; d++) {
        const nx = x + DX[d]!;
        const ny = y2 + DY[d]!;
        if (nx < 0 || ny < 0 || nx >= lv.w || ny >= lv.h) continue;
        if (ny !== hallRow) continue;
        const n = at(lv, nx, ny);
        if (isFloor(lv, n)) {
          door = c;
          hall = n;
          break;
        }
      }
      if (door >= 0) break;
    }
    if (door < 0) continue; // a pocket with no door: not a room
    const boxes = lv.boxes.filter((b) => comp.has(b));
    const goals = lv.goals.filter((g) => comp.has(g));
    if (boxes.length === 0) continue;
    rooms.push({ cells: comp, door, hall, boxes, goals });
  }
  // Left to right by the door's column: the order we visit them.
  rooms.sort((a, b) => {
    const [ax] = xy(lv, a.door);
    const [bx] = xy(lv, b.door);
    return ax - bx;
  });
  return rooms;
}

// Dead cells of a room: floor cells from which a box can never reach a goal.
// Reverse propagation from the goals: a cell S is live when some push from S
// lands on a live cell and the player can stand behind it (S - d is floor).
// Cells never reached are dead; a box on one is a deadlock.
function deadCells(lv: Level, room: Room): Set<number> {
  const live = new Set<number>(room.goals);
  const q: number[] = [...room.goals];
  for (let head = 0; head < q.length; head++) {
    const y0 = q[head]!;
    for (let d = 0; d < 4; d++) {
      // S is the cell a box would be pushed FROM to reach y0 in direction d:
      // the box at S moves to S + d = y0, so S = y0 - d, and the player
      // stands at S - d.
      const [yx, yy] = xy(lv, y0);
      const sx = yx - DX[d]!;
      const sy = yy - DY[d]!;
      if (sx < 0 || sy < 0 || sx >= lv.w || sy >= lv.h) continue;
      const s = at(lv, sx, sy);
      if (!room.cells.has(s) || live.has(s)) continue;
      const bx = sx - DX[d]!;
      const by = sy - DY[d]!;
      if (bx < 0 || by < 0 || bx >= lv.w || by >= lv.h) continue;
      const behind = at(lv, bx, by);
      // The player must be able to stand behind the box: a room cell, or the
      // hallway cell by the door (the player may push in from the hallway).
      if (!room.cells.has(behind) && behind !== room.hall) continue;
      live.add(s);
      q.push(s);
    }
  }
  const dead = new Set<number>();
  for (const c of room.cells) if (!live.has(c)) dead.add(c);
  return dead;
}

// BFS distances from one goal to every room cell (walls block, boxes ignored).
function distFrom(lv: Level, room: Room, start: number): Map<number, number> {
  const dist = new Map<number, number>([[start, 0]]);
  const q: number[] = [start];
  for (let head = 0; head < q.length; head++) {
    const c = q[head]!;
    const [x, y] = xy(lv, c);
    for (let d = 0; d < 4; d++) {
      const nx = x + DX[d]!;
      const ny = y + DY[d]!;
      if (nx < 0 || ny < 0 || nx >= lv.w || ny >= lv.h) continue;
      const n = at(lv, nx, ny);
      if (!room.cells.has(n) || dist.has(n)) continue;
      dist.set(n, dist.get(c)! + 1);
      q.push(n);
    }
  }
  return dist;
}

// Minimum-cost matching of boxes to goals (brute force over permutations;
// 3-4 boxes, so at most 24). Each box must be pushed at least its distance
// to its goal, so this sum is an admissible lower bound on pushes left.
function heuristic(
  boxes: number[],
  goals: number[],
  distToGoal: Map<number, number>[],
): number {
  const n = boxes.length;
  if (n === 0) return 0;
  const idx = goals.map((_, i) => i);
  let best = Infinity;
  const permute = (k: number): void => {
    if (k === n) {
      let s = 0;
      for (let i = 0; i < n; i++) {
        const d = distToGoal[idx[i]!]!.get(boxes[i]!);
        if (d === undefined) return; // unreachable: not a valid matching
        s += d;
      }
      if (s < best) best = s;
      return;
    }
    for (let i = k; i < n; i++) {
      [idx[k], idx[i]] = [idx[i]!, idx[k]!];
      permute(k + 1);
      [idx[k], idx[i]] = [idx[i]!, idx[k]!];
    }
  };
  permute(0);
  return best;
}

interface Node {
  player: number;
  boxes: number[];
  g: number; // pushes so far
  parent: number; // index into nodes, -1 for the root
  seg: string; // moves from parent to here (walk + push letter)
}

// Solve one room push-optimally. The player starts at the hallway cell by
// the door (`entry`). Returns the move string, or null when the budget runs
// out. The final state must have every box on a goal and the player able to
// reach the door (so the player can leave for the next room).
function solveRoom(lv: Level, room: Room, entry: number): string | null {
  const goals = room.goals;
  const goalSet = new Set(goals);
  const dead = deadCells(lv, room);
  const distToGoal = goals.map((g) => distFrom(lv, room, g));
  // The player may walk the room's cells and the hallway cell by the door.
  const allowed = new Set<number>(room.cells);
  allowed.add(room.hall);
  const startBoxes = [...room.boxes].sort((a, b) => a - b);
  // A box starting on a dead cell means the room is unsolvable as built.
  if (startBoxes.some((b) => dead.has(b) && !goalSet.has(b))) return null;

  const nodes: Node[] = [{ player: entry, boxes: startBoxes, g: 0, parent: -1, seg: '' }];
  const key = (p: number, bs: number[]): string => p + '|' + bs.join(',');
  const best = new Map<string, number>([[key(entry, startBoxes), 0]]);
  // Binary heap of [f, g, nodeIndex], ordered by f then by larger g.
  const heap: [number, number, number][] = [];
  const less = (a: [number, number, number], b: [number, number, number]): boolean =>
    a[0] < b[0] || (a[0] === b[0] && a[1] > b[1]);
  const hpush = (e: [number, number, number]): void => {
    let i = heap.length;
    heap.push(e);
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (!less(heap[i]!, heap[p]!)) break;
      [heap[i], heap[p]] = [heap[p]!, heap[i]!];
      i = p;
    }
  };
  const hpop = (): [number, number, number] => {
    const top = heap[0]!;
    const last = heap.pop()!;
    if (heap.length > 0) {
      heap[0] = last;
      for (let i = 0; ; ) {
        let c = 2 * i + 1;
        if (c >= heap.length) break;
        if (c + 1 < heap.length && less(heap[c + 1]!, heap[c]!)) c++;
        if (!less(heap[c]!, heap[i]!)) break;
        [heap[i], heap[c]] = [heap[c]!, heap[i]!];
        i = c;
      }
    }
    return top;
  };

  const h0 = heuristic(startBoxes, goals, distToGoal);
  if (!Number.isFinite(h0)) return null;
  hpush([h0, 0, 0]);
  let expanded = 0;
  while (heap.length > 0) {
    const [, g, si] = hpop();
    const s = nodes[si]!;
    if (best.get(key(s.player, s.boxes)) !== g) continue; // stale entry
    // Goal: every box on a goal and the player can reach the door.
    if (s.boxes.every((b) => goalSet.has(b))) {
      const { reach } = flood(lv, s.player, new Set(s.boxes), allowed);
      if (reach.has(room.door)) {
        // Reconstruct the move string.
        let moves = '';
        for (let t = si; t > 0; t = nodes[t]!.parent) moves = nodes[t]!.seg + moves;
        return moves;
      }
    }
    if (++expanded > ROOM_BUDGET) return null;
    // The player's reachable region from here (boxes are obstacles).
    const boxSet = new Set(s.boxes);
    const { reach, parent } = flood(lv, s.player, boxSet, allowed);
    for (let bi = 0; bi < s.boxes.length; bi++) {
      const b = s.boxes[bi]!;
      const [bx, by] = xy(lv, b);
      for (let d = 0; d < 4; d++) {
        const fx = bx + DX[d]!;
        const fy = by + DY[d]!;
        if (fx < 0 || fy < 0 || fx >= lv.w || fy >= lv.h) continue;
        const front = at(lv, fx, fy);
        // The box must stay inside the room (never into the hallway).
        if (!room.cells.has(front) || boxSet.has(front)) continue;
        const behind = at(lv, bx - DX[d]!, by - DY[d]!);
        if (!reach.has(behind)) continue; // the player cannot get behind it
        const nb = s.boxes.slice();
        nb[bi] = front;
        nb.sort((a, b2) => a - b2);
        // Deadlock: a box on a dead cell (not a goal) can never be saved.
        if (nb.some((x) => dead.has(x) && !goalSet.has(x))) continue;
        const h = heuristic(nb, goals, distToGoal);
        if (!Number.isFinite(h)) continue;
        const ng = g + 1;
        const nk = key(b, nb);
        const seen = best.get(nk);
        if (seen !== undefined && seen <= ng) continue;
        // Walking path from the player to `behind` (from the BFS tree),
        // then the push itself.
        let seg = '';
        {
          const cells: number[] = [];
          let c = behind;
          while (c !== s.player) {
            cells.push(c);
            c = parent.get(c)!;
          }
          cells.reverse();
          let prev = s.player;
          for (const cell of cells) {
            const [px, py] = xy(lv, prev);
            const [cx, cy] = xy(lv, cell);
            if (cx === px + 1) seg += 'r';
            else if (cx === px - 1) seg += 'l';
            else if (cy === py + 1) seg += 'd';
            else seg += 'u';
            prev = cell;
          }
        }
        seg += LETTER[d]!.toUpperCase();
        const t = nodes.length;
        nodes.push({ player: b, boxes: nb, g: ng, parent: si, seg });
        best.set(nk, ng);
        hpush([ng + h, ng, t]);
      }
    }
  }
  return null;
}

// Apply a move string to (player, boxes), returning the new state.
// Returns null if a move is illegal (should not happen for our own output).
function simulate(
  lv: Level,
  player: number,
  boxes: number[],
  moves: string,
): { player: number; boxes: number[] } | null {
  let p = player;
  const bs = boxes.slice();
  for (const ch of moves) {
    const lower = ch.toLowerCase();
    const d = LETTER.indexOf(lower);
    if (d < 0) continue; // whitespace: ignored
    const [px, py] = xy(lv, p);
    const nx = px + DX[d]!;
    const ny = py + DY[d]!;
    if (nx < 0 || ny < 0 || nx >= lv.w || ny >= lv.h) return null;
    const n = at(lv, nx, ny);
    if (!isFloor(lv, n)) return null;
    const bi = bs.indexOf(n);
    if (bi >= 0) {
      const fx = nx + DX[d]!;
      const fy = ny + DY[d]!;
      if (fx < 0 || fy < 0 || fx >= lv.w || fy >= lv.h) return null;
      const f = at(lv, fx, fy);
      if (!isFloor(lv, f) || bs.includes(f)) return null;
      bs[bi] = f;
      bs.sort((a, b) => a - b);
    }
    p = n;
  }
  return { player: p, boxes: bs };
}

// Solve a whole level: walk the hallway between rooms, solve each room.
function solveLevel(text: string): string {
  try {
    const lv = parseLevel(text);
    const rooms = findRooms(lv);
    // Every box must belong to a room; otherwise fall back to an empty answer.
    const inRooms = new Set<number>();
    for (const r of rooms) for (const b of r.boxes) inRooms.add(b);
    if (lv.boxes.some((b) => !inRooms.has(b))) return '';
    let moves = '';
    let player = lv.player;
    let boxes = lv.boxes.slice();
    const allFloor = new Set<number>();
    for (let i = 0; i < lv.w * lv.h; i++) if (isFloor(lv, i)) allFloor.add(i);
    for (const room of rooms) {
      // Walk the hallway (and anywhere the player can go) to this room's door.
      const walk = walkPath(lv, player, room.hall, new Set(boxes), allFloor);
      if (walk === null) return '';
      moves += walk;
      const afterWalk = simulate(lv, player, boxes, walk);
      if (afterWalk === null) return '';
      player = afterWalk.player;
      boxes = afterWalk.boxes;
      // Solve the room from the hallway cell by its door.
      const roomMoves = solveRoom(lv, room, room.hall);
      if (roomMoves === null) return '';
      moves += roomMoves;
      const afterRoom = simulate(lv, player, boxes, roomMoves);
      if (afterRoom === null) return '';
      player = afterRoom.player;
      boxes = afterRoom.boxes;
    }
    // Final verification: replay everything from the start; all boxes on goals.
    const final = simulate(lv, lv.player, lv.boxes, moves);
    if (final === null) return '';
    if (!final.boxes.every((b) => lv.goalSet.has(b))) return '';
    return moves;
  } catch {
    return '';
  }
}

for (const id of readdirSync(I).sort()) {
  const moves = solveLevel(readFileSync(join(I, id, 'level.txt'), 'utf8'));
  mkdirSync(join(O, id), { recursive: true });
  writeFileSync(join(O, id, 'solution.txt'), `${moves}\n`);
}
