#!/bin/bash

if [ -f /app/wfc_main.py ]; then
    cd /app
    streamlit run wfc_main.py --server.port 9900 \
        server.baseURLPath / 2> streamlit.log 
fi


tail -f /dev/null
