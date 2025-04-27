var chatinput = document.getElementById("chatinput");
var lines = document.getElementById("lines");
var loadingbar = document.getElementById("loadingbar");
var linesData = [ ];

socket = opensocket( "/init" );

function submitText() {
    var txt = chatinput.innerText;
    chatinput.innerText = "";

    lines.innerHTML += "<div class='line'>" + txt + "</div>";

    linesData.push( { "role": "user", "content": txt } );
    socket.send( JSON.stringify( linesData ) );
}

function opensocket( url ) {
    socket = new WebSocket( "ws://" + location.host + url );

    socket.addEventListener("open", (event) => { });
    
    socket.addEventListener("close", (event) => { socket = opensocket( "/init" ); });

    socket.addEventListener("message", (event) => processMessage(event) );

    return socket;
}

function processMessage(event) {
    rdata = JSON.parse( event.data );

    if ( rdata.action == "init_system_response" ) {
        loadingbar.style.display = "block";
        lines.innerHTML += "<div class='line server'></div>";
        linesData.push( { "role": "assistant", "content": "" } );
    } else if ( rdata.action == "append_system_response" ) {
        slines = lines.querySelectorAll(".server");
        slines[ slines.length -1 ].innerHTML += rdata.content.replaceAll( "\n", "<br/>" );
        linesData[ linesData.length -1 ].content += rdata.content;
    } else if ( rdata.action == "finish_system_response" ) {
        loadingbar.style.display = "none";
    }
}