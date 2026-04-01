# Knowledge Base

把需要长期检索的资料放在这里，支持 `.md`、`.txt`、`.json`。

## PDF 预处理（MinerU）

项目已支持在重建知识索引时自动调用 MinerU 处理 `knowledge/` 下的 `.pdf` 文件，并把产出的 Markdown 纳入检索索引。

1. 安装并确保命令行可用（示例）：
	- `pip install mineru`
2. 在 `backend/.env` 中配置：
	- `MINERU_ENABLED=true`
	- `MINERU_COMMAND_TEMPLATE=mineru -p {input} -o {output}`
	- `MINERU_TIMEOUT_SECONDS=300`

说明：
- `{input}` 和 `{output}` 是命令模板占位符，会在运行时替换为 PDF 路径和输出目录。
- MinerU 产物会写入 `backend/storage/knowledge/derived/mineru/`，不会污染原始知识库目录。
