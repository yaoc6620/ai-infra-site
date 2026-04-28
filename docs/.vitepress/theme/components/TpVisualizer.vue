<template>
  <div class="tp-viz">
    <div class="tp-controls">
      <button
        :class="['tp-btn', { active: mode === 'attention' }]"
        @click="mode = 'attention'"
      >
        Attention
      </button>
      <button
        :class="['tp-btn', { active: mode === 'mlp' }]"
        @click="mode = 'mlp'"
      >
        MLP
      </button>
      <button class="tp-btn play" @click="playAnimation">
        {{ playing ? 'Reset' : 'Play' }}
      </button>
    </div>

    <!-- Attention mode -->
    <div v-if="mode === 'attention'" class="tp-diagram">
      <div class="tp-step" :class="{ highlight: step >= 0 }">
        <div class="tp-label">Input X <span class="tp-dim">[batch, hidden]</span></div>
        <div class="tp-bar full">X (complete copy on each GPU)</div>
      </div>

      <div class="tp-arrow">Column Parallel (no comm)</div>

      <div class="tp-row" :class="{ highlight: step >= 1 }">
        <div class="tp-gpu">
          <div class="tp-gpu-label">GPU 0</div>
          <div class="tp-bar gpu0">Wqkv[:, :d] &rarr; Q0, K0, V0</div>
          <div class="tp-detail">head 0, 1</div>
        </div>
        <div class="tp-gpu">
          <div class="tp-gpu-label">GPU 1</div>
          <div class="tp-bar gpu1">Wqkv[:, d:] &rarr; Q1, K1, V1</div>
          <div class="tp-detail">head 2, 3</div>
        </div>
      </div>

      <div class="tp-arrow">Independent Attention (no comm)</div>

      <div class="tp-row" :class="{ highlight: step >= 2 }">
        <div class="tp-gpu">
          <div class="tp-gpu-label">GPU 0</div>
          <div class="tp-bar gpu0">Attn(Q0,K0,V0) &rarr; out_0</div>
        </div>
        <div class="tp-gpu">
          <div class="tp-gpu-label">GPU 1</div>
          <div class="tp-bar gpu1">Attn(Q1,K1,V1) &rarr; out_1</div>
        </div>
      </div>

      <div class="tp-arrow">Row Parallel (Wo)</div>

      <div class="tp-row" :class="{ highlight: step >= 3 }">
        <div class="tp-gpu">
          <div class="tp-gpu-label">GPU 0</div>
          <div class="tp-bar gpu0">out_0 @ Wo[:d, :] &rarr; partial_0</div>
        </div>
        <div class="tp-gpu">
          <div class="tp-gpu-label">GPU 1</div>
          <div class="tp-bar gpu1">out_1 @ Wo[d:, :] &rarr; partial_1</div>
        </div>
      </div>

      <div class="tp-allreduce" :class="{ highlight: step >= 4 }">
        AllReduce SUM
      </div>

      <div class="tp-step" :class="{ highlight: step >= 4 }">
        <div class="tp-bar full">Output = partial_0 + partial_1 (complete)</div>
      </div>
    </div>

    <!-- MLP mode -->
    <div v-if="mode === 'mlp'" class="tp-diagram">
      <div class="tp-step" :class="{ highlight: step >= 0 }">
        <div class="tp-label">Input X <span class="tp-dim">[batch, hidden]</span></div>
        <div class="tp-bar full">X (complete copy on each GPU)</div>
      </div>

      <div class="tp-arrow">Column Parallel - gate_up (no comm)</div>

      <div class="tp-row" :class="{ highlight: step >= 1 }">
        <div class="tp-gpu">
          <div class="tp-gpu-label">GPU 0</div>
          <div class="tp-bar gpu0">W_gate_up[:, :d'] &rarr; h_0</div>
          <div class="tp-detail">[batch, intermediate/2]</div>
        </div>
        <div class="tp-gpu">
          <div class="tp-gpu-label">GPU 1</div>
          <div class="tp-bar gpu1">W_gate_up[:, d':] &rarr; h_1</div>
          <div class="tp-detail">[batch, intermediate/2]</div>
        </div>
      </div>

      <div class="tp-arrow">SiLU * gate (element-wise, no comm)</div>

      <div class="tp-row" :class="{ highlight: step >= 2 }">
        <div class="tp-gpu">
          <div class="tp-gpu-label">GPU 0</div>
          <div class="tp-bar gpu0">SiLU(gate_0) * up_0 &rarr; act_0</div>
        </div>
        <div class="tp-gpu">
          <div class="tp-gpu-label">GPU 1</div>
          <div class="tp-bar gpu1">SiLU(gate_1) * up_1 &rarr; act_1</div>
        </div>
      </div>

      <div class="tp-arrow">Row Parallel - down_proj</div>

      <div class="tp-row" :class="{ highlight: step >= 3 }">
        <div class="tp-gpu">
          <div class="tp-gpu-label">GPU 0</div>
          <div class="tp-bar gpu0">act_0 @ W_down[:d', :] &rarr; partial_0</div>
        </div>
        <div class="tp-gpu">
          <div class="tp-gpu-label">GPU 1</div>
          <div class="tp-bar gpu1">act_1 @ W_down[d':, :] &rarr; partial_1</div>
        </div>
      </div>

      <div class="tp-allreduce" :class="{ highlight: step >= 4 }">
        AllReduce SUM
      </div>

      <div class="tp-step" :class="{ highlight: step >= 4 }">
        <div class="tp-bar full">Output = partial_0 + partial_1 (complete)</div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'

const mode = ref('attention')
const step = ref(-1)
const playing = ref(false)
let timer = null

function playAnimation() {
  if (playing.value) {
    clearInterval(timer)
    playing.value = false
    step.value = -1
    return
  }
  playing.value = true
  step.value = 0
  timer = setInterval(() => {
    if (step.value >= 4) {
      clearInterval(timer)
      playing.value = false
      return
    }
    step.value++
  }, 800)
}
</script>

<style scoped>
.tp-viz {
  border: 1px solid var(--vp-c-divider);
  border-radius: 12px;
  padding: 24px;
  margin: 20px 0;
  background: var(--vp-c-bg-soft);
}

.tp-controls {
  display: flex;
  gap: 8px;
  margin-bottom: 20px;
}

.tp-btn {
  padding: 6px 16px;
  border: 1px solid var(--vp-c-divider);
  border-radius: 8px;
  background: var(--vp-c-bg);
  cursor: pointer;
  font-size: 0.9em;
  transition: all 0.2s;
  color: var(--vp-c-text-1);
}

.tp-btn:hover {
  border-color: var(--vp-c-brand-1);
}

.tp-btn.active {
  background: var(--vp-c-brand-1);
  color: white;
  border-color: var(--vp-c-brand-1);
}

.tp-btn.play {
  margin-left: auto;
  background: #22c55e;
  color: white;
  border-color: #22c55e;
}

.tp-diagram {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.tp-step {
  opacity: 0.4;
  transition: opacity 0.4s;
}

.tp-step.highlight {
  opacity: 1;
}

.tp-label {
  font-size: 0.85em;
  color: var(--vp-c-text-2);
  margin-bottom: 4px;
}

.tp-dim {
  font-family: monospace;
  font-size: 0.85em;
}

.tp-bar {
  padding: 10px 16px;
  border-radius: 8px;
  font-family: monospace;
  font-size: 0.85em;
  text-align: center;
}

.tp-bar.full {
  background: linear-gradient(135deg, #ede9fe, #dbeafe);
  border: 1px solid #c7d2fe;
  color: #1e1b4b;
}

.dark .tp-bar.full {
  background: linear-gradient(135deg, #2e1065, #172554);
  border: 1px solid #4338ca;
  color: #e0e7ff;
}

.tp-bar.gpu0 {
  background: #ede9fe;
  border: 1px solid #c4b5fd;
  color: #4c1d95;
}

.dark .tp-bar.gpu0 {
  background: #2e1065;
  border: 1px solid #6d28d9;
  color: #c4b5fd;
}

.tp-bar.gpu1 {
  background: #dbeafe;
  border: 1px solid #93c5fd;
  color: #1e3a5f;
}

.dark .tp-bar.gpu1 {
  background: #172554;
  border: 1px solid #2563eb;
  color: #93c5fd;
}

.tp-row {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 12px;
  opacity: 0.4;
  transition: opacity 0.4s;
}

.tp-row.highlight {
  opacity: 1;
}

.tp-gpu {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.tp-gpu-label {
  font-size: 0.8em;
  font-weight: 600;
  color: var(--vp-c-text-2);
}

.tp-detail {
  font-size: 0.75em;
  color: var(--vp-c-text-3);
  font-family: monospace;
}

.tp-arrow {
  text-align: center;
  font-size: 0.8em;
  color: var(--vp-c-text-3);
  padding: 4px 0;
  position: relative;
}

.tp-arrow::before {
  content: '↓ ';
}

.tp-allreduce {
  text-align: center;
  padding: 10px;
  background: #fef3c7;
  border: 2px solid #f59e0b;
  border-radius: 24px;
  font-weight: 700;
  color: #92400e;
  font-size: 0.95em;
  opacity: 0.4;
  transition: all 0.4s;
}

.dark .tp-allreduce {
  background: #451a03;
  border-color: #d97706;
  color: #fcd34d;
}

.tp-allreduce.highlight {
  opacity: 1;
  box-shadow: 0 0 20px rgba(245, 158, 11, 0.3);
}
</style>
