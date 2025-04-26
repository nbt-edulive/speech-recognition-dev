// Biến toàn cục
let currentSessionId = null;
let isRecording = false;
let mediaRecorder = null;
let audioChunks = [];
let currentAudioUrl = null;

// DOM Elements
const recordBtn = document.getElementById('record-btn');
const recordingStatus = document.getElementById('recording-status');
const chatContainer = document.getElementById('chat-container');
const sessionList = document.getElementById('session-list');
const newSessionBtn = document.getElementById('new-session-btn');
const voiceSelect = document.getElementById('voice-select');
const statusIndicator = document.getElementById('status-indicator');
const audioControls = document.getElementById('audio-controls');
const playBtn = document.getElementById('play-btn');
const audioPlayer = document.getElementById('audio-player');
const emptyState = document.getElementById('empty-state');
const textInput = document.getElementById('text-input');
const sendBtn = document.getElementById('send-btn');

// Event Listeners
document.addEventListener('DOMContentLoaded', function() {
    // Khởi tạo phiên mới nếu không có phiên nào
    if (sessionList.children.length === 0) {
        createNewSession();
    } else {
        // Chọn phiên đầu tiên
        const firstSession = sessionList.querySelector('.session-item');
        if (firstSession) {
            firstSession.click();
        }
    }

    // Thêm sự kiện click cho nút tạo phiên mới
    if (newSessionBtn) {
        newSessionBtn.addEventListener('click', createNewSession);
    }

    // Thêm sự kiện cho input text
    if (textInput) {
        textInput.addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                sendTextMessage();
            }
        });
    }

    // Thêm sự kiện cho nút gửi
    if (sendBtn) {
        sendBtn.addEventListener('click', sendTextMessage);
    }
});

// Thêm sự kiện click cho nút ghi âm
if (recordBtn) {
    recordBtn.addEventListener('click', toggleRecording);
}

// Thêm sự kiện click cho nút phát lại
if (playBtn) {
    playBtn.addEventListener('click', playAudio);
}

// Xử lý các sự kiện cho phiên chat
if (sessionList) {
    sessionList.addEventListener('click', function(e) {
        // Xử lý nút xóa
        if (e.target.classList.contains('delete-btn')) {
            const sessionId = e.target.getAttribute('data-session-id');
            deleteSession(sessionId);
            e.stopPropagation();
            return;
        }
        
        // Xử lý khi nhấp vào phiên
        if (e.target.classList.contains('session-item')) {
            const sessionId = e.target.getAttribute('data-session-id');
            loadSession(sessionId);
        }
    });
}

// Functions
async function toggleRecording() {
    if (!isRecording) {
        startRecording();
    } else {
        stopRecording();
    }
}

async function startRecording() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        
        mediaRecorder = new MediaRecorder(stream);
        audioChunks = [];
        
        mediaRecorder.ondataavailable = (e) => {
            audioChunks.push(e.data);
        };
        
        mediaRecorder.onstop = processRecording;
        
        mediaRecorder.start();
        isRecording = true;
        
        // UI updates
        recordBtn.classList.add('recording');
        recordingStatus.textContent = 'Đang ghi âm...';
        showStatus('Đang ghi âm...', 'info');
    } catch (err) {
        console.error('Không thể truy cập microphone:', err);
        showStatus('Không thể truy cập microphone. Vui lòng kiểm tra quyền truy cập.', 'danger');
    }
}

function stopRecording() {
    if (mediaRecorder && isRecording) {
        mediaRecorder.stop();
        isRecording = false;
        
        // UI updates
        recordBtn.classList.remove('recording');
        recordingStatus.textContent = 'Đang xử lý...';
        showStatus('Đang xử lý âm thanh...', 'info');
    }
}

async function processRecording() {
    try {
        // Tạo file audio từ chunks
        const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
        
        // Tạo form data để gửi lên server
        const formData = new FormData();
        formData.append('audio', audioBlob, 'recording.wav');
        formData.append('session_id', currentSessionId || 'new');
        formData.append('voice', voiceSelect.value);
        
        showStatus('Đang gửi lên server...', 'info');
        
        // Gửi lên server để xử lý
        const response = await fetch('/api/process-audio', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (data.success) {
            // Cập nhật session ID nếu là phiên mới
            if (!currentSessionId || currentSessionId === 'new') {
                currentSessionId = data.session_id;
                // Cập nhật danh sách phiên nếu cần
                updateSessionList();
            }
            
            // Hiển thị tin nhắn mới
            addMessage('user', data.user_text);
            addMessage('assistant', data.assistant_response);
            
            // Lưu URL audio để phát lại
            currentAudioUrl = data.audio_url;
            audioPlayer.src = data.audio_url;
            audioControls.style.display = 'flex';
            
            // Tự động phát audio
            playAudio();
            
            showStatus('Đã xử lý thành công!', 'success');
            recordingStatus.textContent = 'Nhấn để ghi âm';
        } else {
            throw new Error(data.error || 'Có lỗi xảy ra');
        }
    } catch (error) {
        console.error('Lỗi khi xử lý âm thanh:', error);
        showStatus('Lỗi: ' + error.message, 'danger');
        recordingStatus.textContent = 'Nhấn để ghi âm';
    }
}

// Hàm xử lý gửi tin nhắn text
async function sendTextMessage() {
    const text = textInput.value.trim();
    
    if (!text) {
        return;
    }
    
    try {
        showStatus('Đang xử lý...', 'info');
        
        // Chuẩn bị dữ liệu gửi đi
        const payload = {
            text: text,
            session_id: currentSessionId || 'new',
            voice: voiceSelect.value
        };
        
        // Xóa nội dung input và disable nút gửi
        textInput.value = '';
        textInput.disabled = true;
        sendBtn.disabled = true;
        
        // Gửi request lên server
        const response = await fetch('/api/process-text', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(payload)
        });
        
        const data = await response.json();
        
        if (data.success) {
            // Cập nhật session ID nếu là phiên mới
            if (!currentSessionId || currentSessionId === 'new') {
                currentSessionId = data.session_id;
                updateSessionList();
            }
            
            // Hiển thị tin nhắn mới
            addMessage('user', data.user_text);
            addMessage('assistant', data.assistant_response);
            
            // Lưu URL audio để phát lại
            currentAudioUrl = data.audio_url;
            audioPlayer.src = data.audio_url;
            audioControls.style.display = 'flex';
            
            // Tự động phát audio
            playAudio();
            
            showStatus('Đã xử lý thành công!', 'success');
        } else {
            throw new Error(data.error || 'Có lỗi xảy ra');
        }
    } catch (error) {
        console.error('Lỗi khi xử lý tin nhắn text:', error);
        showStatus('Lỗi: ' + error.message, 'danger');
    } finally {
        // Enable lại input và nút gửi
        textInput.disabled = false;
        sendBtn.disabled = false;
        textInput.focus();
    }
}

function addMessage(role, content) {
    // Xóa thông báo trống nếu có
    if (emptyState) {
        emptyState.style.display = 'none';
    }
    
    const messageDiv = document.createElement('div');
    messageDiv.classList.add('message');
    messageDiv.classList.add(role === 'user' ? 'user-message' : 'assistant-message');
    
    const contentP = document.createElement('p');
    contentP.textContent = content;
    messageDiv.appendChild(contentP);
    
    const timeSpan = document.createElement('span');
    timeSpan.classList.add('message-time');
    timeSpan.textContent = new Date().toLocaleTimeString();
    messageDiv.appendChild(timeSpan);
    
    chatContainer.appendChild(messageDiv);
    
    // Cuộn xuống tin nhắn mới nhất
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

async function loadSession(sessionId) {
    try {
        // Cập nhật UI
        const allSessions = document.querySelectorAll('.session-item');
        allSessions.forEach(s => s.classList.remove('active'));
        document.querySelector(`.session-item[data-session-id="${sessionId}"]`).classList.add('active');
        
        showStatus('Đang tải phiên chat...', 'info');
        
        // Gọi API lấy lịch sử
        const response = await fetch(`/api/session/${sessionId}`);
        const data = await response.json();
        
        if (data.success) {
            // Cập nhật session ID hiện tại
            currentSessionId = sessionId;
            
            // Xóa tất cả tin nhắn hiện tại
            chatContainer.innerHTML = '';
            
            // Nếu không có tin nhắn, hiển thị trạng thái trống
            if (data.messages.length === 0) {
                chatContainer.innerHTML = `
                    <div class="text-center text-muted empty-state" id="empty-state">
                        <h4>Chào mừng đến với Trợ lý Gia sư AI</h4>
                        <p>Nhấn nút microphone hoặc nhập câu hỏi để bắt đầu trò chuyện</p>
                    </div>
                `;
            } else {
                // Hiển thị các tin nhắn
                data.messages.forEach(msg => {
                    addMessageFromHistory(msg);
                });
            }
            
            showStatus('Đã tải phiên chat thành công!', 'success');
            setTimeout(() => {
                statusIndicator.style.display = 'none';
            }, 2000);
        } else {
            throw new Error(data.error || 'Có lỗi xảy ra');
        }
    } catch (error) {
        console.error('Lỗi khi tải phiên chat:', error);
        showStatus('Lỗi: ' + error.message, 'danger');
    }
}

function addMessageFromHistory(message) {
    const messageDiv = document.createElement('div');
    messageDiv.classList.add('message');
    messageDiv.classList.add(message.role === 'user' ? 'user-message' : 'assistant-message');
    
    const contentP = document.createElement('p');
    contentP.textContent = message.content;
    messageDiv.appendChild(contentP);
    
    const time = new Date(message.timestamp).toLocaleTimeString();
    const timeSpan = document.createElement('span');
    timeSpan.classList.add('message-time');
    timeSpan.textContent = time;
    messageDiv.appendChild(timeSpan);
    
    chatContainer.appendChild(messageDiv);
    
    // Cập nhật URL audio cho tin nhắn cuối cùng từ assistant
    if (message.role === 'assistant' && message.audio_path) {
        currentAudioUrl = message.audio_path;
        audioPlayer.src = message.audio_path;
        audioControls.style.display = 'flex';
    }
}

async function createNewSession() {
    try {
        showStatus('Đang tạo phiên mới...', 'info');
        
        const response = await fetch('/api/session', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                title: `Hội thoại ${new Date().toLocaleString()}`
            })
        });
        
        const data = await response.json();
        
        if (data.success) {
            // Cập nhật danh sách phiên
            updateSessionList();
            
            // Tải phiên mới
            setTimeout(() => {
                loadSession(data.session_id);
            }, 500);
            
            showStatus('Đã tạo phiên mới thành công!', 'success');
        } else {
            throw new Error(data.error || 'Có lỗi xảy ra');
        }
    } catch (error) {
        console.error('Lỗi khi tạo phiên mới:', error);
        showStatus('Lỗi: ' + error.message, 'danger');
    }
}

async function deleteSession(sessionId) {
    if (!confirm('Bạn có chắc chắn muốn xóa phiên chat này?')) {
        return;
    }
    
    try {
        showStatus('Đang xóa phiên chat...', 'info');
        
        const response = await fetch(`/api/session/${sessionId}`, {
            method: 'DELETE'
        });
        
        const data = await response.json();
        
        if (data.success) {
            // Xóa phiên khỏi danh sách
            const sessionElement = document.querySelector(`.session-item[data-session-id="${sessionId}"]`);
            if (sessionElement) {
                sessionElement.remove();
            }
            
            // Nếu xóa phiên hiện tại, chuyển sang phiên khác hoặc tạo mới
            if (currentSessionId === sessionId) {
                const firstSession = sessionList.querySelector('.session-item');
                if (firstSession) {
                    loadSession(firstSession.getAttribute('data-session-id'));
                } else {
                    createNewSession();
                }
            }
            
            showStatus('Đã xóa phiên chat thành công!', 'success');
        } else {
            throw new Error(data.error || 'Có lỗi xảy ra');
        }
    } catch (error) {
        console.error('Lỗi khi xóa phiên chat:', error);
        showStatus('Lỗi: ' + error.message, 'danger');
    }
}

async function updateSessionList() {
    try {
        // Get all sessions
        const response = await fetch('/');
        const html = await response.text();
        
        // Create a DOM parser
        const parser = new DOMParser();
        const doc = parser.parseFromString(html, 'text/html');
        
        // Extract the session list HTML
        const newSessionList = doc.getElementById('session-list').innerHTML;
        sessionList.innerHTML = newSessionList;
    } catch (error) {
        console.error('Lỗi khi cập nhật danh sách phiên:', error);
    }
}

function playAudio() {
    if (audioPlayer.src) {
        audioPlayer.play();
    }
}

function showStatus(message, type) {
    statusIndicator.textContent = message;
    statusIndicator.className = 'status-indicator';
    statusIndicator.classList.add(`bg-${type}`);
    statusIndicator.classList.add('text-white');
    statusIndicator.style.display = 'block';
    
    if (type === 'success' || type === 'danger') {
        setTimeout(() => {
            statusIndicator.style.display = 'none';
        }, 3000);
    }
}