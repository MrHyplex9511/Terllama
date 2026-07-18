import type { Message, ChatSession } from '../../types';

let messages = $state<Message[]>([]);
let isGenerating = $state(false);
let currentResponse = $state('');
let tokensPerSecond = $state(0);
let sessions = $state<ChatSession[]>([]);
let activeSessionId = $state<string | null>(null);

function generateId(): string {
  return 'chat-' + Date.now().toString(36) + Math.random().toString(36).slice(2, 8);
}

export function getChatState() {
  return {
    get messages() {
      return messages;
    },
    get isGenerating() {
      return isGenerating;
    },
    get currentResponse() {
      return currentResponse;
    },
    get tokensPerSecond() {
      return tokensPerSecond;
    },
    get sessions() {
      return sessions;
    },
    get activeSessionId() {
      return activeSessionId;
    },
    setMessages,
    setIsGenerating,
    setCurrentResponse,
    setTokensPerSecond,
    setSessions,
    setActiveSessionId,
    newSession,
    switchSession,
    deleteSession,
    addMessage,
    loadSessions,
    saveSessions,
  };
}

function setMessages(v: Message[]) {
  messages = v;
}
function setIsGenerating(v: boolean) {
  isGenerating = v;
}
function setCurrentResponse(v: string) {
  currentResponse = v;
}
function setTokensPerSecond(v: number) {
  tokensPerSecond = v;
}
function setSessions(v: ChatSession[]) {
  sessions = v;
}
function setActiveSessionId(v: string | null) {
  activeSessionId = v;
}

function newSession() {
  const id = generateId();
  const session: ChatSession = {
    id,
    title: 'New Chat',
    messages: [],
    created_at: new Date().toISOString(),
  };
  sessions = [...sessions, session];
  activeSessionId = id;
  messages = [];
  saveSessions();
}

function switchSession(id: string) {
  const session = sessions.find((s) => s.id === id);
  if (session) {
    activeSessionId = id;
    messages = session.messages;
  }
}

function deleteSession(id: string) {
  sessions = sessions.filter((s) => s.id !== id);
  if (activeSessionId === id) {
    activeSessionId = sessions.length > 0 ? sessions[0].id : null;
    messages = activeSessionId
      ? sessions.find((s) => s.id === activeSessionId)?.messages ?? []
      : [];
  }
  saveSessions();
}

function addMessage(msg: Message) {
  messages = [...messages, msg];
  // Update session
  sessions = sessions.map((s) => {
    if (s.id === activeSessionId) {
      const updated = { ...s, messages: [...messages] };
      if (updated.title === 'New Chat' && messages.length > 0 && messages[0].role === 'user') {
        updated.title = messages[0].content.slice(0, 50) + (messages[0].content.length > 50 ? '...' : '');
      }
      return updated;
    }
    return s;
  });
  saveSessions();
}

function loadSessions() {
  try {
    const data = localStorage.getItem('terllama-sessions');
    if (data) {
      sessions = JSON.parse(data);
    }
  } catch {
    sessions = [];
  }
}

function saveSessions() {
  try {
    localStorage.setItem('terllama-sessions', JSON.stringify(sessions));
  } catch {
    // ignore storage errors
  }
}
