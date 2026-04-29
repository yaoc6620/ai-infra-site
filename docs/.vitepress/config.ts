import { defineConfig } from 'vitepress'
import { withMermaid } from 'vitepress-plugin-mermaid'
import mathjax3 from 'markdown-it-mathjax3'

export default withMermaid(
  defineConfig({
    title: 'AI Infra Interview Guide',
    description: 'AI Infra 知识复习 & 面试复盘',
    lang: 'zh-CN',
    base: '/ai-infra-site/',

    markdown: {
      config: (md) => {
        md.use(mathjax3)
      },
    },

    vue: {
      template: {
        compilerOptions: {
          isCustomElement: (tag) => tag.startsWith('mjx-'),
        },
      },
    },

    themeConfig: {
      nav: [
        { text: '首页', link: '/' },
        { text: '知识库', link: '/knowledge/' },
        { text: '面试复盘', link: '/interviews/' },
      ],

      sidebar: {
        '/knowledge/': [
          {
            text: '并行策略',
            collapsed: false,
            items: [
              { text: 'Tensor Parallelism', link: '/knowledge/tensor-parallelism' },
              { text: 'Pipeline Parallelism', link: '/knowledge/pipeline-parallelism' },
              { text: 'Data Parallelism', link: '/knowledge/data-parallelism' },
            ],
          },
          {
            text: '推理引擎',
            collapsed: false,
            items: [
              { text: 'vLLM 架构', link: '/knowledge/inference-engine' },
              { text: 'KV Cache', link: '/knowledge/kv-cache' },
              { text: 'Attention 优化', link: '/knowledge/attention-optimization' },
              { text: 'FlashAttention', link: '/knowledge/flash-attention' },
            ],
          },
          {
            text: '底层基础',
            collapsed: false,
            items: [
              { text: 'CUDA 编程', link: '/knowledge/cuda-basics' },
            ],
          },
        ],
        '/interviews/': [
          {
            text: '面试复盘',
            items: [
              { text: '复盘总览', link: '/interviews/' },
              { text: 'WXG 暑期实习 一面', link: '/interviews/2026-04-wx-wxg-1st' },
              { text: 'WXG 暑期实习 二面', link: '/interviews/2026-04-wx-wxg-2nd' },
              { text: '模板示例', link: '/interviews/2026-04-xx-example' },
            ],
          },
        ],
      },

      search: {
        provider: 'local',
      },

      outline: {
        level: [2, 3],
        label: '目录',
      },

      socialLinks: [
        { icon: 'github', link: 'https://github.com' },
      ],

      footer: {
        message: 'AI Infra Interview Prep',
      },
    },

    mermaid: {},
  })
)
