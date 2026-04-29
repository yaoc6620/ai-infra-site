<template>
  <div v-if="!authenticated" class="pw-overlay">
    <div class="pw-card">
      <h2>AI Infra Interview Guide</h2>
      <p>Please enter password to continue</p>
      <form @submit.prevent="checkPassword">
        <input
          v-model="password"
          type="password"
          placeholder="Password"
          class="pw-input"
          autofocus
        />
        <button type="submit" class="pw-btn">Enter</button>
      </form>
      <p v-if="error" class="pw-error">Incorrect password</p>
    </div>
  </div>
  <slot v-else />
</template>

<script setup>
import { ref, onMounted } from 'vue'

const PASSWORD_HASH = '85c7501603a35fa7de2c65c7b41e4944a3f85c26ae76577ba5d4989a3c3d2ca8'

const authenticated = ref(false)
const password = ref('')
const error = ref(false)

async function sha256(message) {
  const msgBuffer = new TextEncoder().encode(message)
  const hashBuffer = await crypto.subtle.digest('SHA-256', msgBuffer)
  const hashArray = Array.from(new Uint8Array(hashBuffer))
  return hashArray.map(b => b.toString(16).padStart(2, '0')).join('')
}

async function checkPassword() {
  error.value = false
  const hash = await sha256(password.value)
  if (hash === PASSWORD_HASH) {
    authenticated.value = true
    sessionStorage.setItem('ai-infra-auth', 'true')
  } else {
    error.value = true
  }
}

onMounted(() => {
  if (sessionStorage.getItem('ai-infra-auth') === 'true') {
    authenticated.value = true
  }
})
</script>

<style scoped>
.pw-overlay {
  position: fixed;
  inset: 0;
  z-index: 9999;
  display: flex;
  align-items: center;
  justify-content: center;
  background: var(--vp-c-bg);
}

.pw-card {
  text-align: center;
  padding: 48px;
  border: 1px solid var(--vp-c-divider);
  border-radius: 16px;
  background: var(--vp-c-bg-soft);
  max-width: 400px;
  width: 90%;
}

.pw-card h2 {
  margin: 0 0 8px 0;
  background: linear-gradient(135deg, #6366f1, #ec4899);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}

.pw-card p {
  margin: 0 0 24px 0;
  color: var(--vp-c-text-2);
  font-size: 0.9em;
}

.pw-input {
  display: block;
  width: 100%;
  padding: 10px 16px;
  border: 1px solid var(--vp-c-divider);
  border-radius: 8px;
  font-size: 1em;
  background: var(--vp-c-bg);
  color: var(--vp-c-text-1);
  margin-bottom: 12px;
  box-sizing: border-box;
}

.pw-input:focus {
  outline: none;
  border-color: var(--vp-c-brand-1);
}

.pw-btn {
  width: 100%;
  padding: 10px;
  border: none;
  border-radius: 8px;
  background: var(--vp-c-brand-1);
  color: white;
  font-size: 1em;
  cursor: pointer;
  transition: background 0.2s;
}

.pw-btn:hover {
  background: var(--vp-c-brand-2);
}

.pw-error {
  color: #ef4444 !important;
  margin-top: 12px !important;
  margin-bottom: 0 !important;
}
</style>
