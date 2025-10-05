# Latex2MathJax Plugin for Obsidian

When you copy math content from large language models (like ChatGPT), the formulas are typically provided in LaTeX format. However, Obsidian supports MathJax, and this plugin will **automatically convert the LaTeX formulas into MathJax format** when pasted into your notes.

[中文版本](./README.zh-cn.md)

## Features

- **Automatic Replacement on Paste:**  
  Automatically replaces LaTeX math delimiters when content is pasted into the editor.
  
- **Manual Replacement:**  
  Use commands to replace LaTeX math delimiters in selected text or the entire document.

## Installation

1. Go to the Release page ([https://github.com/aqpower/obsidian-Latex2MathJax/releases](https://github.com/aqpower/obsidian-Latex2MathJax/releases)) and download the latest version of the plugin in `zip` format.
2. Extract the downloaded plugin files and place them into Obsidian's plugin folder `.obsidian/plugins`.
3. Reload Obsidian, then enable the plugin in the Third-party Plugins section.

## Usage

1. Paste LaTeX formulas into the editor.
2. Open the command palette (`Ctrl/Cmd + P`) and search for `Latex2MathJax` to use the desired functionality.
3. Alternatively, enable the auto-replace option in the plugin settings, so the plugin will automatically replace LaTeX formulas when you paste them.
