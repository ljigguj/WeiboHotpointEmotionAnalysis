@echo off
echo 启动微博热点舆情分析系统...

:: 新窗口运行爬虫（爬完自动关闭）
start "热搜评论爬虫" cmd /k "python StaManu/HotpointSrape.py"

:: 新窗口运行 Streamlit 面板（保持运行）
start "舆情分析面板" cmd /k "streamlit run dashboard.py"

echo 两个窗口已启动，请查看弹出的终端。
