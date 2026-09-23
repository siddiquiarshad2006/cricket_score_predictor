mkdir -p ~/.streamlit/

echo "\
[general]\n\
email = \"\"\n\
\n\
[server]\n\
port = $PORT\n\
enableCORS = false\n\
enableXsrfProtection = false\n\
headless = true\n\
\n\
[browser]\n\
gatherUsageStats = false\n\
" > ~/.streamlit/config.toml