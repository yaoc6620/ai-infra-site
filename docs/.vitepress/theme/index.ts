import DefaultTheme from 'vitepress/theme'
import TpVisualizer from './components/TpVisualizer.vue'
import './custom.css'

export default {
  extends: DefaultTheme,
  enhanceApp({ app }) {
    app.component('TpVisualizer', TpVisualizer)
  },
}
