import enum

class AudienceType(str, enum.Enum):
    management = "management"
    experts = "experts"
    investors = "investors"

class SlideVisualType(str, enum.Enum):
    text = "text"
    chart = "chart"
    table = "table"
    image = "image"

class SlideStatus(str, enum.Enum):
    draft = "draft"
    generating = "generating"
    ready = "ready"
    error = "error"

class JobType(str, enum.Enum):
    generate_slide = "generate_slide"
    export_pptx = "export_pptx"
    export_pdf = "export_pdf"

class JobStatus(str, enum.Enum):
    queued = "queued"
    running = "running"
    done = "done"
    error = "error"

class FileKind(str, enum.Enum):
    export_pptx = "export_pptx"
    export_pdf = "export_pdf"
    image = "image"
    document = "document"
    template = "template"
