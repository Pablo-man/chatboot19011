var chatinput = document.getElementById("chatinput");
var lines = document.getElementById("lines");
var loadingbar = document.getElementById("loadingbar");
var linesData = [];
var currentCaseId = null;

// Añadir nuevo campo para el modo
var currentMode = "normal"; // "normal", "case_waiting_user", "comparison"

// Conexión WebSocket
socket = opensocket("/init");

// Función para mostrar el panel de entrada de caso
function showCaseInput() {
    document.getElementById("caseInputPanel").style.display = "block";
    document.getElementById("userResponsePanel").style.display = "none";
    document.getElementById("normalChatPanel").style.display = "none";
}

// Función para mostrar el panel de respuesta de usuario
function showUserResponseInput() {
    document.getElementById("caseInputPanel").style.display = "none";
    document.getElementById("userResponsePanel").style.display = "block";
    document.getElementById("normalChatPanel").style.display = "none";
}

// Función para mostrar el panel de chat normal
function showNormalChatInput() {
    document.getElementById("caseInputPanel").style.display = "none";
    document.getElementById("userResponsePanel").style.display = "none";
    document.getElementById("normalChatPanel").style.display = "block";
}

// Función para enviar un nuevo caso
function submitCase() {
    var caseText = document.getElementById("caseInputText").value;
    if (!caseText.trim()) return;
    
    // Generar ID único para este caso
    currentCaseId = "case_" + Date.now();
    
    // Mostrar el caso en la interfaz
    lines.innerHTML += `<div class="line case-study">
        <h3>Caso de Estudio:</h3>
        <div>${caseText}</div>
    </div>`;
    
    // Indicar que estamos procesando
    loadingbar.style.display = "block";
    
    // Enviar al servidor
    socket.send(JSON.stringify({
        action: "new_case",
        case_id: currentCaseId,
        content: caseText
    }));
    
    // Limpiar campo
    document.getElementById("caseInputText").value = "";
    
    // Cambiar modo
    currentMode = "case_waiting_user";
}

// Función para enviar respuesta del usuario
function submitUserResponse() {
    var userResponse = document.getElementById("userResponseText").value;
    if (!userResponse.trim() || !currentCaseId) return;
    
    // Mostrar la respuesta del usuario en la interfaz
    lines.innerHTML += `<div class="line user-response">
        <h3>Tu Respuesta:</h3>
        <div>${userResponse}</div>
    </div>`;
    
    // Indicar que estamos procesando
    loadingbar.style.display = "block";
    
    // Enviar al servidor
    socket.send(JSON.stringify({
        action: "submit_user_response",
        case_id: currentCaseId,
        content: userResponse
    }));
    
    // Limpiar campo
    document.getElementById("userResponseText").value = "";
    
    // Cambiar modo
    currentMode = "comparison";
}

// Función original modificada para compatibilidad
function submitText() {
    var txt = chatinput.innerText;
    if (!txt.trim()) return false;
    
    chatinput.innerText = "";
    
    lines.innerHTML += "<div class='line'>" + txt + "</div>";
    
    linesData.push({ "role": "user", "content": txt });
    socket.send(JSON.stringify(linesData));
    
    return false;
}

function opensocket(url) {
    socket = new WebSocket("wss://" + location.host + url);
    
    socket.addEventListener("open", (event) => {});
    
    socket.addEventListener("close", (event) => { 
        socket = opensocket("/init"); 
    });
    
    socket.addEventListener("message", (event) => processMessage(event));
    
    return socket;
}

function processMessage(event) {
    rdata = JSON.parse(event.data);
    
    if (rdata.action == "init_system_response") {
        loadingbar.style.display = "block";
        lines.innerHTML += "<div class='line server'></div>";
        linesData.push({ "role": "assistant", "content": "" });
    } 
    else if (rdata.action == "append_system_response") {
        slines = lines.querySelectorAll(".server");
        slines[slines.length - 1].innerHTML += rdata.content.replaceAll("\n", "<br/>");
        linesData[linesData.length - 1].content += rdata.content;
    } 
    else if (rdata.action == "finish_system_response") {
        loadingbar.style.display = "none";
    }
    else if (rdata.action == "case_processed") {
        // Caso procesado, mostrar panel de respuesta del usuario
        loadingbar.style.display = "none";
        lines.innerHTML += `<div class="line system-message">
            Caso procesado. Por favor proporciona tu respuesta.
        </div>`;
        showUserResponseInput();
    }
    else if (rdata.action == "ai_response") {
        // Mostrar la respuesta del AI
        lines.innerHTML += `<div class="line ai-response">
            <h3>Respuesta del Asistente:</h3>
            <div>${marked.parse(rdata.content.replaceAll("\n", "<br/>"))}</div>
        </div>`;
    }
    else if (rdata.action == "comparison") {
        // Mostrar la comparación
        lines.innerHTML += `<div class="line comparison">
            <h3>Análisis Comparativo:</h3>
            <div>${marked.parse(rdata.content.replaceAll("\n", "<br/>"))}</div>
        </div>`;
        
        // Mostrar botón para nuevo caso
        lines.innerHTML += `<div class="line system-message">
            <button onclick="resetAndStartNewCase()" class="new-case-btn">Iniciar Nuevo Caso</button>
        </div>`;
        
        loadingbar.style.display = "none";
        showCaseInput(); // Mostrar panel de nuevo caso
    }
    else if (rdata.action == "error") {
        lines.innerHTML += `<div class="line error">
            Error: ${rdata.message}
        </div>`;
        loadingbar.style.display = "none";
    }
    
    // Desplazar hacia abajo para ver el último mensaje
    linescontainer = document.getElementById("linescontainer");
    linescontainer.scrollTop = linescontainer.scrollHeight;
}

// Función para reiniciar y comenzar un nuevo caso
function resetAndStartNewCase() {
    currentCaseId = null;
    currentMode = "normal";
    showCaseInput();
}

// Cambiar a modo normal (chat regular)
function switchToNormalMode() {
    currentMode = "normal";
    showNormalChatInput();
}

// Cambiar a modo caso de estudio
function switchToCaseMode() {
    showCaseInput();
}

// Inicialización - mostrar panel de caso por defecto
document.addEventListener("DOMContentLoaded", function() {
    showCaseInput();
});