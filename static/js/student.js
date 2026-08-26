const ws = new WebSocket(`ws://${window.SIGNALING_HOST}/ws/student`);
const teacherVideo = document.getElementById('teacherVideo');
const classroomVideo = document.getElementById('classroomVideo');
let peerConnection;
const config = { iceServers: [{ urls: 'stun:stun.l.google.com:19302' }] };

async function initWebRTC() {
    try {
        const localStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
        classroomVideo.srcObject = localStream;

        peerConnection = new RTCPeerConnection(config);
        localStream.getTracks().forEach(track => peerConnection.addTrack(track, localStream));

        peerConnection.ontrack = (event) => {
            teacherVideo.srcObject = event.streams[0];
        };

        peerConnection.onicecandidate = (event) => {
            if (event.candidate) {
                ws.send(JSON.stringify({ candidate: event.candidate }));
            }
        };

        ws.onmessage = async (event) => {
            const data = JSON.parse(event.data);
            if (data.offer) {
                await peerConnection.setRemoteDescription(new RTCSessionDescription(data.offer));
                const answer = await peerConnection.createAnswer();
                await peerConnection.setLocalDescription(answer);
                ws.send(JSON.stringify({ answer: answer }));
            } else if (data.candidate) {
                await peerConnection.addIceCandidate(new RTCIceCandidate(data.candidate));
            }
        };
    } catch (err) {
        console.error("Error initializing camera stream on Pi.", err);
    }
}

initWebRTC();