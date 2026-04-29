<template>
  <PasswordGuard>
    <DefaultLayout />
  </PasswordGuard>
</template>

<script setup>
import DefaultTheme from 'vitepress/theme'
import PasswordGuard from './components/PasswordGuard.vue'

const DefaultLayout = DefaultTheme.Layout
</script>
