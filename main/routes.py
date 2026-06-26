from flask import Blueprint, render_template

main_bp = Blueprint(
    "main",
    __name__,
    template_folder="../templates",
    static_folder="../static",
)


@main_bp.route("/")
def index():
    features = [
        {
            "icon": "bi-search",
            "title": "Smart Research",
            "description": (
                "Automatically retrieve and summarise relevant academic "
                "papers, articles, and web sources for any topic."
            ),
        },
        {
            "icon": "bi-file-earmark-text",
            "title": "Document Analysis",
            "description": (
                "Upload PDFs or documents and extract key insights, "
                "themes, and citations with one click."
            ),
        },
        {
            "icon": "bi-diagram-3",
            "title": "Knowledge Graph",
            "description": (
                "Visualise connections between concepts, authors, and "
                "sources to uncover hidden relationships."
            ),
        },
        {
            "icon": "bi-bar-chart-line",
            "title": "Data Analytics",
            "description": (
                "Run quantitative analyses on your research data and "
                "generate publication-ready charts instantly."
            ),
        },
        {
            "icon": "bi-robot",
            "title": "AI Assistance",
            "description": (
                "Ask questions in plain language and receive grounded, "
                "cited answers from your document corpus."
            ),
        },
        {
            "icon": "bi-download",
            "title": "Report Export",
            "description": (
                "Export polished research reports in PDF, Word, or "
                "Markdown format at any stage of your workflow."
            ),
        },
    ]
    return render_template("index.html", features=features)
