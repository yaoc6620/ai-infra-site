import DefaultTheme from 'vitepress/theme'
import Layout from './Layout.vue'
import TpVisualizer from './components/TpVisualizer.vue'
import './custom.css'

export default {
  extends: DefaultTheme,
  Layout,
  enhanceApp({ app }) {
    app.component('TpVisualizer', TpVisualizer)
  },
}
