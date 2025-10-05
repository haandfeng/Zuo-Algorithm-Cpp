import type { App, Editor } from 'obsidian'
import { Notice, Plugin, PluginSettingTab, Setting } from 'obsidian'

// 定义插件的设置接口
interface Latex2MathJaxSettings {
  autoReplaceOnPaste: boolean // 是否启用自动替换粘贴内容
}

const DEFAULT_SETTINGS: Latex2MathJaxSettings = {
  autoReplaceOnPaste: false, // 默认不启用自动替换
}

// 定义插件的主类
export default class Latex2MathJax extends Plugin {
  settings: Latex2MathJaxSettings
  pasteEventListener: ((evt: ClipboardEvent, editor: Editor) => void) | null = null

  async onload() {
    // 加载插件设置
    await this.loadSettings()

    // 注册设置面板
    this.addSettingTab(new Latex2MathJaxSettingTab(this.app, this))

    // 监听设置变化
    this.applyPasteEventListener()

    // 注册命令：替换选中的内容
    this.addCommand({
      id: 'replace-latex-math-markers',
      name: '替换选中内容中的 Latex 标记为 MathJax 标记',
      editorCallback: (editor: Editor) => {
        const selection = editor.getSelection()
        if (selection) {
          const updatedText = this.replaceMathDelimitersInText(selection)
          editor.replaceSelection(updatedText)
        }
      },
    })

    // 注册命令： 替换整个文档中的Latex标记
    this.addCommand({
      id: 'replace-all-latex-math-markers',
      name: '替换文档中的 Latex 标记为 MathJax 标记',
      editorCallback: (editor: Editor) => {
        const content = editor.getValue()
        const updatedContent = this.replaceMathDelimitersInText(content)
        editor.setValue(updatedContent)
      },
    })
  }

  // 根据设置决定是否启用自动替换
  applyPasteEventListener() {
    if (this.pasteEventListener) {
      this.app.workspace.off('editor-paste', this.pasteEventListener)
    }

    // 如果启用了自动替换，注册粘贴事件监听器
    if (this.settings.autoReplaceOnPaste) {
      this.pasteEventListener = (evt: ClipboardEvent, editor: Editor) => {
        const clipboardData = evt.clipboardData?.getData('text/plain')
        if (clipboardData) {
          const updatedContent = this.replaceMathDelimitersInText(clipboardData)
          if (updatedContent !== clipboardData) {
            evt.preventDefault()
            editor.replaceSelection(updatedContent)
          }
        }
      }
      this.app.workspace.on('editor-paste', this.pasteEventListener)
    }
    else {
      this.pasteEventListener = null // 清空事件监听器
    }
  }

  // 替换函数：替换文本中的数学标记
  replaceMathDelimitersInText(content: string): string {
    let updatedContent = content

    // 第一步：替换包含空格的标记
    updatedContent = updatedContent
      .replace(/\\\(\s/g, '$') // 替换 "\( " 为 "$"
      .replace(/\s\\\)/g, '$') // 替换 " \)" 为 "$"
      .replace(/\\\[\s/g, '$$$$') // 替换 "\[ " 为 "$$$$" (等于 $$)
      .replace(/\s\\\]/g, '$$$$') // 替换 " \]" 为 "$$$$" (等于 $$)

    // 第二步：替换没有空格的标记
    updatedContent = updatedContent
      .replace(/\\\(/g, '$') // 替换 "\(" 为 "$"
      .replace(/\\\)/g, '$') // 替换 "\)" 为 "$"
      .replace(/\\\[/g, '$$$$') // 替换 "\[" 为 "$$"
      .replace(/\\\]/g, '$$$$') // 替换 "\]" 为 "$$"

    // 检查文本是否发生了变化
    if (content !== updatedContent) {
      // 如果发生了替换，发出通知
      new Notice('😊 成功替换为 MathJax 格式~')
    }

    return updatedContent
  }

  // 加载插件设置
  async loadSettings() {
    this.settings = Object.assign({}, DEFAULT_SETTINGS, await this.loadData())
    this.applyPasteEventListener() // 应用当前设置
  }

  // 保存插件设置
  async saveSettings() {
    await this.saveData(this.settings)
    this.applyPasteEventListener() // 应用当前设置
  }
}

// 设置面板
class Latex2MathJaxSettingTab extends PluginSettingTab {
  plugin: Latex2MathJax

  constructor(app: App, plugin: Latex2MathJax) {
    super(app, plugin)
    this.plugin = plugin
  }

  display(): void {
    const { containerEl } = this

    containerEl.empty()

    // 启用自动替换选项
    new Setting(containerEl)
      .setName('启用自动替换粘贴内容中的数学标记')
      .setDesc('启用后，粘贴的内容中的 LaTeX 数学标记会被自动替换为 MathJax 格式。')
      .addToggle(toggle => toggle
        .setValue(this.plugin.settings.autoReplaceOnPaste)
        .onChange(async (value) => {
          this.plugin.settings.autoReplaceOnPaste = value
          await this.plugin.saveSettings()
        }))
  }
}
