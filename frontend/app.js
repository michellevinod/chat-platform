// ============================================================
// Document Intelligence Frontend
// ============================================================

const API_BASE = "http://127.0.0.1:8000";

let currentProject = "";
let currentDocument = "";
let sessionId = crypto.randomUUID();
let conversationId = sessionId;


// ============================================================
// DOM HELPERS
// ============================================================

function $(id) {
    return document.getElementById(id);
}


function showToast(message, isError = false) {
    const toast = $("toast");

    if (!toast) {
        return;
    }

    toast.textContent = message;
    toast.classList.remove("hidden");

    if (isError) {
        toast.classList.add("error");
    } else {
        toast.classList.remove("error");
    }

    setTimeout(() => {
        toast.classList.add("hidden");
    }, 3500);
}


function escapeHtml(value) {
    if (value === null || value === undefined) {
        return "";
    }

    return String(value)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}


function setUploadStatus(message, isError = false) {
    const element = $("uploadStatus");

    if (!element) {
        return;
    }

    element.textContent = message;
    element.classList.remove("hidden");

    if (isError) {
        element.classList.add("error");
    } else {
        element.classList.remove("error");
    }
}


// ============================================================
// API
// ============================================================

async function apiRequest(path, options = {}) {
    const response = await fetch(`${API_BASE}${path}`, options);

    const text = await response.text();

    let data = {};

    if (text) {
        try {
            data = JSON.parse(text);
        } catch {
            data = {
                detail: text
            };
        }
    }

    if (!response.ok) {
        let message = "Request failed.";

        if (typeof data.detail === "string") {
            message = data.detail;
        } else if (Array.isArray(data.detail)) {
            message = data.detail
                .map(item => item.msg || JSON.stringify(item))
                .join(", ");
        } else if (data.message) {
            message = data.message;
        }

        throw new Error(message);
    }

    return data;
}


// ============================================================
// PROJECTS
// ============================================================

async function loadProjects() {
    const select = $("projectSelect");

    if (!select) {
        return;
    }

    try {
        select.innerHTML = `<option value="">Loading projects...</option>`;

        const data = await apiRequest("/projects");

        const projects = Array.isArray(data)
            ? data
            : (data.projects || []);

        select.innerHTML = "";

        if (!projects.length) {
            select.innerHTML =
                `<option value="">No projects found</option>`;

            currentProject = "";
            clearDocuments();

            return;
        }

        select.innerHTML =
            `<option value="">Select project</option>`;

        projects.forEach(project => {
            let name = "";

            if (typeof project === "string") {
                name = project;
            } else if (project && project.name) {
                name = project.name;
            }

            if (!name) {
                return;
            }

            const option = document.createElement("option");
            option.value = name;
            option.textContent = name;

            select.appendChild(option);
        });

        // Restore currently selected project if it still exists.
        if (currentProject) {
            const exists = Array.from(select.options)
                .some(option => option.value === currentProject);

            if (exists) {
                select.value = currentProject;
                await loadDocuments();
            }
        }

    } catch (error) {
        console.error("Load projects error:", error);

        select.innerHTML =
            `<option value="">Unable to load projects</option>`;

        showToast(
            `Could not load projects: ${error.message}`,
            true
        );
    }
}


// ============================================================
// DOCUMENTS
// ============================================================

function clearDocuments() {
    const list = $("documentList");

    if (list) {
        list.innerHTML = `
            <div class="empty-state">
                Select a project
            </div>
        `;
    }

    currentDocument = "";

    updateDocumentHeader();
}


async function loadDocuments() {
    const list = $("documentList");

    if (!list) {
        return;
    }

    if (!currentProject) {
        clearDocuments();
        return;
    }

    try {
        list.innerHTML = `
            <div class="empty-state">
                Loading documents...
            </div>
        `;

        const path =
            `/documents?project_name=${encodeURIComponent(currentProject)}`;

        const data = await apiRequest(path);

        let documents = [];

        if (Array.isArray(data)) {
            documents = data;
        } else if (Array.isArray(data.documents)) {
            documents = data.documents;
        } else if (Array.isArray(data.results)) {
            documents = data.results;
        }

        renderDocuments(documents);

    } catch (error) {
        console.error("Load documents error:", error);

        list.innerHTML = `
            <div class="empty-state">
                Unable to load documents
            </div>
        `;

        showToast(
            `Could not load documents: ${error.message}`,
            true
        );
    }
}


function normalizeDocumentName(document) {
    if (typeof document === "string") {
        return document;
    }

    if (!document) {
        return "";
    }

    return (
        document.document_name ||
        document.name ||
        document.filename ||
        document.file_name ||
        ""
    );
}


function renderDocuments(documents) {
    const list = $("documentList");

    if (!list) {
        return;
    }

    list.innerHTML = "";

    if (!documents.length) {
        list.innerHTML = `
            <div class="empty-state">
                No documents in this project
            </div>
        `;

        currentDocument = "";
        updateDocumentHeader();

        return;
    }

    documents.forEach(document => {
        const name = normalizeDocumentName(document);

        if (!name) {
            return;
        }

        const button = documentElement("button");

        button.className = "document-item";

        if (name === currentDocument) {
            button.classList.add("active");
        }

        button.textContent = name;
        button.title = name;

        button.addEventListener("click", () => {
            selectDocument(name);
        });

        list.appendChild(button);
    });
}


function documentElement(tag) {
    return window.document.createElement(tag);
}


function selectDocument(documentName) {
    currentDocument = documentName;

    const items = document.querySelectorAll(".document-item");

    items.forEach(item => {
        item.classList.toggle(
            "active",
            item.textContent === documentName
        );
    });

    updateDocumentHeader();

    clearChat();

    showToast(`Selected document: ${documentName}`);
}


function updateDocumentHeader() {
    const title = $("currentDocumentTitle");
    const subtitle = $("currentDocumentSubtitle");

    if (!currentProject) {
        if (title) {
            title.textContent = "Document Assistant";
        }

        if (subtitle) {
            subtitle.textContent =
                "Select a project and document to begin";
        }

        return;
    }

    if (!currentDocument) {
        if (title) {
            title.textContent = currentProject;
        }

        if (subtitle) {
            subtitle.textContent =
                "Select a document to begin";
        }

        return;
    }

    if (title) {
        title.textContent = currentDocument;
    }

    if (subtitle) {
        subtitle.textContent =
            `Project: ${currentProject}`;
    }
}


// ============================================================
// UPLOAD
// ============================================================

async function uploadDocument(file) {
    if (!file) {
        return;
    }

    if (!currentProject) {
        showToast(
            "Please select a project before uploading.",
            true
        );

        return;
    }

    const allowedExtensions = [
        ".pdf",
        ".docx",
        ".pptx",
        ".xlsx"
    ];

    const fileName = file.name.toLowerCase();

    const valid = allowedExtensions.some(
        extension => fileName.endsWith(extension)
    );

    if (!valid) {
        showToast(
            "Please upload a PDF, DOCX, PPTX or XLSX file.",
            true
        );

        return;
    }

    try {
        setUploadStatus(
            `Uploading ${file.name}...`
        );

        const formData = new FormData();

        formData.append("file", file);
        formData.append("project_name", currentProject);
        formData.append("document_name", file.name);

        const data = await apiRequest("/upload", {
            method: "POST",
            body: formData
        });

        console.log("Upload response:", data);

        setUploadStatus(
            `${file.name} uploaded successfully.`
        );

        showToast("Document uploaded successfully.");

        await loadDocuments();

        // Automatically select uploaded document.
        currentDocument = file.name;

        renderDocuments(
            await getDocumentsForCurrentProject()
        );

        selectDocument(file.name);

    } catch (error) {
        console.error("Upload error:", error);

        setUploadStatus(
            `Upload failed: ${error.message}`,
            true
        );

        showToast(
            `Upload failed: ${error.message}`,
            true
        );
    }
}


async function getDocumentsForCurrentProject() {
    if (!currentProject) {
        return [];
    }

    try {
        const path =
            `/documents?project_name=${encodeURIComponent(currentProject)}`;

        const data = await apiRequest(path);

        if (Array.isArray(data)) {
            return data;
        }

        if (Array.isArray(data.documents)) {
            return data.documents;
        }

        if (Array.isArray(data.results)) {
            return data.results;
        }

        return [];

    } catch (error) {
        console.error("Get documents error:", error);
        return [];
    }
}


// ============================================================
// DELETE
// ============================================================

function openDeleteModal(documentName) {
    const modal = $("deleteModal");
    const text = $("deleteModalText");

    if (!modal) {
        return;
    }

    currentDocument = documentName;

    if (text) {
        text.textContent =
            `Delete "${documentName}" from project "${currentProject}"?`;
    }

    modal.classList.remove("hidden");
}


function closeDeleteModal() {
    const modal = $("deleteModal");

    if (modal) {
        modal.classList.add("hidden");
    }
}


async function deleteDocument() {
    if (!currentProject || !currentDocument) {
        closeDeleteModal();
        return;
    }

    const project = currentProject;
    const documentName = currentDocument;

    try {
        const path =
            `/documents?project_name=${encodeURIComponent(project)}` +
            `&document_name=${encodeURIComponent(documentName)}`;

        const data = await apiRequest(path, {
            method: "DELETE"
        });

        console.log("Delete response:", data);

        closeDeleteModal();

        currentDocument = "";

        clearChat();

        await loadDocuments();

        showToast("Document deleted successfully.");

    } catch (error) {
        console.error("Delete document error:", error);

        showToast(
            `Could not delete document: ${error.message}`,
            true
        );
    }
}


// ============================================================
// CHAT
// ============================================================

function clearChat() {
    const container = $("chatContainer");

    if (!container) {
        return;
    }

    container.innerHTML = `
        <div
            id="welcomeMessage"
            class="welcome"
        >

            <div class="welcome-icon">
                ✦
            </div>

            <h2>
                Ask anything about your document
            </h2>

            <p>
                Search through text, tables, figures and images
                using natural language.
            </p>

            <div class="example-prompts">

                <button
                    class="prompt-button"
                    data-prompt="Summarize this document"
                >
                    Summarize this document
                </button>

                <button
                    class="prompt-button"
                    data-prompt="Show me the important tables in this document"
                >
                    Show important tables
                </button>

                <button
                    class="prompt-button"
                    data-prompt="Show me the figures in this document"
                >
                    Show figures
                </button>

            </div>

        </div>
    `;

    attachPromptButtons();
}


function attachPromptButtons() {
    document.querySelectorAll(".prompt-button").forEach(button => {
        button.addEventListener("click", () => {
            const prompt = button.dataset.prompt || "";

            const input = $("chatInput");

            if (!input) {
                return;
            }

            input.value = prompt;
            input.focus();
        });
    });
}


function appendMessage(role, content, citations = []) {
    const container = $("chatContainer");

    if (!container) {
        return;
    }

    const welcome = $("welcomeMessage");

    if (welcome) {
        welcome.remove();
    }

    const message = document.createElement("div");

    message.className =
        role === "user"
            ? "chat-message user-message"
            : "chat-message assistant-message";

    const bubble = document.createElement("div");

    bubble.className = "message-bubble";

    if (role === "assistant") {
        if (typeof marked !== "undefined") {
            try {
                bubble.innerHTML = marked.parse(content || "");
            } catch {
                bubble.textContent = content || "";
            }
        } else {
            bubble.textContent = content || "";
        }
    } else {
        bubble.textContent = content || "";
    }

    message.appendChild(bubble);

    if (
        role === "assistant" &&
        Array.isArray(citations) &&
        citations.length
    ) {
        const citationElement =
            renderCitations(citations);

        message.appendChild(citationElement);
    }

    container.appendChild(message);

    container.scrollTop = container.scrollHeight;
}


function renderCitations(citations) {
    const wrapper = document.createElement("div");

    wrapper.className = "citations";

    const heading = document.createElement("div");

    heading.className = "citation-heading";
    heading.textContent = "Sources";

    wrapper.appendChild(heading);

    citations.forEach((citation, index) => {
        if (!citation) {
            return;
        }

        const item = document.createElement("div");

        item.className = "citation-item";

        const documentName =
            citation.document || "Unknown document";

        const page =
            citation.page !== undefined
                ? citation.page
                : "?";

        const chunkType =
            citation.chunk_type || "";

        item.innerHTML = `
            <span class="citation-number">
                ${index + 1}
            </span>

            <div class="citation-content">
                <strong>
                    ${escapeHtml(documentName)}
                </strong>

                <span>
                    Page ${escapeHtml(page)}
                    ${chunkType
                        ? ` · ${escapeHtml(chunkType)}`
                        : ""}
                </span>
            </div>
        `;

        wrapper.appendChild(item);
    });

    return wrapper;
}


function appendLoadingMessage() {
    const container = $("chatContainer");

    if (!container) {
        return null;
    }

    const message = document.createElement("div");

    message.className =
        "chat-message assistant-message loading-message";

    message.innerHTML = `
        <div class="message-bubble">
            <span class="typing-indicator">
                Thinking...
            </span>
        </div>
    `;

    container.appendChild(message);

    container.scrollTop = container.scrollHeight;

    return message;
}


async function sendMessage() {
    const input = $("chatInput");
    const button = $("sendButton");

    if (!input) {
        return;
    }

    const query = input.value.trim();

    if (!query) {
        return;
    }

    if (!currentProject) {
        showToast(
            "Please select a project first.",
            true
        );

        return;
    }

    if (!currentDocument) {
        showToast(
            "Please select a document first.",
            true
        );

        return;
    }

    input.value = "";

    appendMessage(
        "user",
        query
    );

    const loadingMessage =
        appendLoadingMessage();

    if (button) {
        button.disabled = true;
    }

    input.disabled = true;

    try {
        const payload = {
            query: query,
            project_name: currentProject,
            document_name: currentDocument,
            session_id: sessionId,
            conversation_id: conversationId
        };

        const data = await apiRequest("/chat", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(payload)
        });

        if (loadingMessage) {
            loadingMessage.remove();
        }

        if (data.session_id) {
            sessionId = data.session_id;
            conversationId = data.session_id;
        }

        const response =
            data.response ||
            "I couldn't generate a response.";

        appendMessage(
            "assistant",
            response,
            data.citations || []
        );

    } catch (error) {
        console.error("Chat error:", error);

        if (loadingMessage) {
            loadingMessage.remove();
        }

        appendMessage(
            "assistant",
            `Sorry, I couldn't process that request.\n\n**Error:** ${error.message}`
        );

    } finally {
        if (button) {
            button.disabled = false;
        }

        input.disabled = false;
        input.focus();
    }
}


// ============================================================
// INITIALIZATION
// ============================================================

function setupProjectSelection() {
    const select = $("projectSelect");

    if (!select) {
        return;
    }

    select.addEventListener("change", async () => {
        currentProject = select.value;
        currentDocument = "";

        updateDocumentHeader();
        clearChat();

        if (currentProject) {
            await loadDocuments();
        } else {
            clearDocuments();
        }
    });
}


function setupUpload() {
    const uploadArea = $("uploadArea");
    const fileInput = $("fileInput");

    if (!uploadArea || !fileInput) {
        return;
    }

    uploadArea.addEventListener("click", () => {
        if (!currentProject) {
            showToast(
                "Please select a project before uploading.",
                true
            );

            return;
        }

        fileInput.click();
    });

    fileInput.addEventListener("change", async () => {
        const file = fileInput.files?.[0];

        if (file) {
            await uploadDocument(file);
        }

        fileInput.value = "";
    });

    uploadArea.addEventListener("dragover", event => {
        event.preventDefault();
        uploadArea.classList.add("dragover");
    });

    uploadArea.addEventListener("dragleave", () => {
        uploadArea.classList.remove("dragover");
    });

    uploadArea.addEventListener("drop", async event => {
        event.preventDefault();

        uploadArea.classList.remove("dragover");

        if (!currentProject) {
            showToast(
                "Please select a project before uploading.",
                true
            );

            return;
        }

        const file =
            event.dataTransfer?.files?.[0];

        if (file) {
            await uploadDocument(file);
        }
    });
}


function setupRefreshDocuments() {
    const button = $("refreshDocumentsBtn");

    if (!button) {
        return;
    }

    button.addEventListener("click", async () => {
        if (!currentProject) {
            showToast(
                "Please select a project first.",
                true
            );

            return;
        }

        await loadDocuments();

        showToast("Documents refreshed.");
    });
}


function setupChat() {
    const input = $("chatInput");
    const button = $("sendButton");

    if (button) {
        button.addEventListener("click", sendMessage);
    }

    if (input) {
        input.addEventListener("keydown", event => {
            if (event.key === "Enter" && !event.shiftKey) {
                event.preventDefault();
                sendMessage();
            }
        });

        input.addEventListener("input", () => {
            input.style.height = "auto";
            input.style.height =
                `${Math.min(input.scrollHeight, 160)}px`;
        });
    }

    attachPromptButtons();
}


function setupDelete() {
    const cancelButton =
        $("cancelDeleteButton");

    const confirmButton =
        $("confirmDeleteButton");

    if (cancelButton) {
        cancelButton.addEventListener(
            "click",
            closeDeleteModal
        );
    }

    if (confirmButton) {
        confirmButton.addEventListener(
            "click",
            deleteDocument
        );
    }
}


async function checkApiConnection() {
    try {
        await apiRequest("/health");

        console.log("Backend API connected.");
    } catch (error) {
        console.warn(
            "Backend API health check failed:",
            error
        );

        showToast(
            "Backend API is not reachable. Make sure FastAPI is running.",
            true
        );
    }
}


document.addEventListener("DOMContentLoaded", async () => {
    console.log("Document Intelligence frontend starting...");

    setupProjectSelection();
    setupUpload();
    setupRefreshDocuments();
    setupChat();
    setupDelete();

    updateDocumentHeader();

    await checkApiConnection();
    await loadProjects();
});