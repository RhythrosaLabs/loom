 <style>
    /* Background and Text Color */
    .reportview-container {
        background-color: #1a1a1a;
        color: white;
    }
    .sidebar .sidebar-content {
        background-color: #333333;
    }
    
    /* Header Styling */
    h1 {
        color: #FFD700;
        text-align: center;
        font-family: 'Helvetica', sans-serif;
    }
    
    /* Button Styling */
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        height: 3em;
        width: 15em;
        border-radius:10px;
        border: 1px solid #4CAF50;
        font-size:20px;
    }
    
    .stButton>button:hover {
        background-color: #45a049;
    }
    
    /* Success and Error Message Styling */
    .css-1aumxhk.edgvbvh3 {
        background-color: #1a1a1a;
    }
    
    /* Tooltip Styling */
    div.tooltip {
        position: relative;
        display: inline-block;
        border-bottom: 1px dotted black;
    }
    
    div.tooltip .tooltiptext {
        visibility: hidden;
        width: 200px;
        background-color: #555;
        color: #fff;
        text-align: center;
        border-radius: 6px;
        padding: 5px 0;
        position: absolute;
        z-index: 1;
        bottom: 125%; 
        left: 50%;
        margin-left: -100px;
        opacity: 0;
        transition: opacity 0.3s;
    }
    
    div.tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
    
    /* Grid Layout for Images */
    .image-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
        grid-gap: 10px;
    }
    
    /* Footer Styling */
    footer {
        visibility: hidden;
    }
    </style>
    """, unsafe_allow_html=True)
