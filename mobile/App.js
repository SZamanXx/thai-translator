import React, { useState, useRef, useEffect, useCallback } from 'react';
import {
  StyleSheet,
  Text,
  View,
  Pressable,
  FlatList,
  ActivityIndicator,
  TextInput,
  Alert,
  KeyboardAvoidingView,
  Platform,
  LogBox,
} from 'react-native';

// Suppress annoying Expo SDK 54 warning
LogBox.ignoreLogs([
  'Could not access feature flag',
  'disableEventLoopOnBridgeless',
]);
import { StatusBar } from 'expo-status-bar';
import { Audio } from 'expo-av';
import * as FileSystem from 'expo-file-system';
import * as Linking from 'expo-linking';
import AsyncStorage from '@react-native-async-storage/async-storage';
import QRCode from 'react-native-qrcode-svg';

// ==================== CONFIGURATION ====================
// Set to your ngrok / cloudflared / public backend URL.
// Example: 'https://abc123.ngrok-free.app'
const BACKEND_URL = '';
const WS_URL = BACKEND_URL.replace('https://', 'wss://').replace('http://', 'ws://') + '/ws';

// VAD Settings (threshold is now dynamic, stored in state)
const DEFAULT_SILENCE_THRESHOLD_DB = -35;
const MIN_THRESHOLD_DB = -50;  // Most sensitive
const MAX_THRESHOLD_DB = -15;  // Least sensitive (ignores most noise)
const SILENCE_DURATION_MS = 1500;
const MIN_SPEECH_DURATION_MS = 500;

export default function App() {
  // ==================== AUTH STATE ====================
  const [authToken, setAuthToken] = useState(null);
  const [loggedInUser, setLoggedInUser] = useState('');
  const [authLoading, setAuthLoading] = useState(true);
  const [authUsername, setAuthUsername] = useState('');
  const [authPassword, setAuthPassword] = useState('');

  // ==================== NAVIGATION STATE ====================
  const [screen, setScreen] = useState('loading'); // 'loading', 'auth', 'login', 'register', 'mode', 'solo', 'roomSetup', 'waiting', 'join', 'room'
  const [pendingRoomCode, setPendingRoomCode] = useState(null); // From deep link

  // ==================== USER STATE ====================
  const [myName, setMyName] = useState('');
  const [partnerName, setPartnerName] = useState('');
  const [roomCode, setRoomCode] = useState('');
  const [isHost, setIsHost] = useState(false);

  // ==================== CONVERSATION STATE ====================
  const [isListening, setIsListening] = useState(false);
  const [currentDb, setCurrentDb] = useState(-60);
  const [silenceThreshold, setSilenceThreshold] = useState(DEFAULT_SILENCE_THRESHOLD_DB);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const [messages, setMessages] = useState([]);
  const [status, setStatus] = useState('');

  // ==================== REFS ====================
  const wsRef = useRef(null);
  const soundRef = useRef(null);
  const recordingRef = useRef(null);
  const audioQueueRef = useRef([]);
  const isPlayingRef = useRef(false);
  const isRecordingRef = useRef(false);
  const isListeningRef = useRef(false);
  const speechStartTimeRef = useRef(null);
  const silenceStartTimeRef = useRef(null);
  const flatListRef = useRef(null);
  const reconnectTimeoutRef = useRef(null);

  // ==================== AUTH CHECK ON STARTUP ====================
  useEffect(() => {
    const checkAuth = async () => {
      try {
        // Load saved threshold
        const savedThreshold = await AsyncStorage.getItem('silenceThreshold');
        if (savedThreshold) {
          setSilenceThreshold(parseInt(savedThreshold, 10));
        }

        const token = await AsyncStorage.getItem('authToken');
        const username = await AsyncStorage.getItem('username');

        if (token && username) {
          // Validate token with backend
          const response = await fetch(`${BACKEND_URL}/validate`, {
            headers: {
              'Authorization': `Bearer ${token}`,
              'ngrok-skip-browser-warning': 'true',
            },
          });

          if (response.ok) {
            setAuthToken(token);
            setLoggedInUser(username);
            setMyName(username);
            setScreen('mode');
          } else {
            // Token invalid - clear and show auth
            await AsyncStorage.removeItem('authToken');
            await AsyncStorage.removeItem('username');
            setScreen('auth');
          }
        } else {
          setScreen('auth');
        }
      } catch (err) {
        console.log('Auth check error:', err);
        setScreen('auth');
      } finally {
        setAuthLoading(false);
      }
    };
    checkAuth();
  }, []);

  // ==================== DEEP LINK HANDLING ====================
  useEffect(() => {
    const checkInitialUrl = async () => {
      const url = await Linking.getInitialURL();
      if (url) handleDeepLink(url);
    };
    checkInitialUrl();

    const subscription = Linking.addEventListener('url', ({ url }) => {
      handleDeepLink(url);
    });

    return () => subscription.remove();
  }, []);

  const handleDeepLink = (url) => {
    try {
      const { path } = Linking.parse(url);
      if (path && path.startsWith('join/')) {
        const code = path.split('/')[1];
        if (code && code.length === 4) {
          setPendingRoomCode(code);
          setScreen('join');
        }
      }
    } catch (e) {
      console.error('Deep link error:', e);
    }
  };

  // ==================== AUDIO SETUP ====================
  useEffect(() => {
    const setupAudio = async () => {
      try {
        const { status } = await Audio.requestPermissionsAsync();
        if (status !== 'granted') {
          Alert.alert('Permission Required', 'This app needs microphone access');
          return;
        }
        await Audio.setAudioModeAsync({
          allowsRecordingIOS: true,
          playsInSilentModeIOS: true,
          staysActiveInBackground: false,
        });
      } catch (err) {
        console.error('Audio setup error:', err);
      }
    };
    setupAudio();

    return () => {
      if (soundRef.current) soundRef.current.unloadAsync();
      if (wsRef.current) wsRef.current.close();
      if (reconnectTimeoutRef.current) clearTimeout(reconnectTimeoutRef.current);
    };
  }, []);

  // ==================== WEBSOCKET ====================
  const connectWebSocket = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    const ws = new WebSocket(WS_URL);

    ws.onopen = () => {
      console.log('WebSocket connected');
      setStatus('Connected');
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        handleWsMessage(data);
      } catch (e) {
        console.error('WS parse error:', e);
      }
    };

    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
      setStatus('Connection error');
    };

    ws.onclose = () => {
      console.log('WebSocket closed');
      if (screen === 'room' || screen === 'waiting') {
        setStatus('Disconnected - reconnecting...');
        reconnectTimeoutRef.current = setTimeout(connectWebSocket, 2000);
      }
    };

    wsRef.current = ws;
  }, [screen]);

  const handleWsMessage = async (data) => {
    switch (data.type) {
      case 'room_created':
        setRoomCode(data.code);
        setScreen('waiting');
        setStatus('Waiting for partner...');
        break;

      case 'user_joined':
        setPartnerName(data.name);
        setScreen('room');
        setStatus('Partner joined!');
        break;

      case 'room_ready':
        setPartnerName(data.partner);
        setRoomCode(data.code);
        setScreen('room');
        setStatus('Connected to partner');
        break;

      case 'transcript':
        const isMe = data.from === myName;
        const newMsg = {
          id: `msg-${data.timestamp}`,
          from: data.from,
          isMe,
          original: data.original,
          translated: data.translated,
          langFrom: data.lang_from,
          langTo: data.lang_to,
          timestamp: data.timestamp,
        };
        setMessages(prev => [...prev, newMsg]);
        break;

      case 'audio_for_you':
        audioQueueRef.current.push(data.data);
        playNextInQueue();
        break;

      case 'user_left':
        Alert.alert('Partner left the room', `${data.name} disconnected`);
        leaveRoom();
        break;

      case 'error':
        Alert.alert('Error', data.message);
        break;
    }
  };

  const sendWsMessage = (msg) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(msg));
    }
  };

  // ==================== ROOM MANAGEMENT ====================
  const createRoom = () => {
    if (!myName.trim()) {
      Alert.alert('Error', 'Enter your name');
      return;
    }
    setIsHost(true);
    connectWebSocket();
    setTimeout(() => {
      sendWsMessage({ type: 'create_room', name: myName.trim(), token: authToken });
    }, 500);
  };

  const joinRoom = (code) => {
    if (!myName.trim()) {
      Alert.alert('Error', 'Enter your name');
      return;
    }
    const roomToJoin = code || pendingRoomCode || roomCode;
    if (!roomToJoin || roomToJoin.length !== 4) {
      Alert.alert('Error', 'Enter valid room code (4 digits)');
      return;
    }
    setIsHost(false);
    setRoomCode(roomToJoin);
    connectWebSocket();
    setTimeout(() => {
      sendWsMessage({ type: 'join_room', code: roomToJoin, name: myName.trim() });
    }, 500);
  };

  const leaveRoom = () => {
    sendWsMessage({ type: 'leave_room' });
    if (wsRef.current) wsRef.current.close();
    wsRef.current = null;
    setScreen('mode');
    setMessages([]);
    setRoomCode('');
    setPartnerName('');
    setIsListening(false);
    isListeningRef.current = false;
    setPendingRoomCode(null);
    setStatus('');
  };

  // ==================== AUTH FUNCTIONS ====================
  const handleRegister = async () => {
    if (!authUsername.trim() || !authPassword.trim()) {
      Alert.alert('Error', 'Enter username and password');
      return;
    }

    try {
      setAuthLoading(true);
      const response = await fetch(`${BACKEND_URL}/register`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'ngrok-skip-browser-warning': 'true',
        },
        body: JSON.stringify({
          username: authUsername.trim(),
          password: authPassword,
        }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || 'Registration failed');
      }

      // Save token and go to mode screen
      await AsyncStorage.setItem('authToken', data.token);
      await AsyncStorage.setItem('username', data.username);
      setAuthToken(data.token);
      setLoggedInUser(data.username);
      setMyName(data.username);
      setAuthUsername('');
      setAuthPassword('');
      setScreen('mode');
    } catch (err) {
      Alert.alert('Error', err.message);
    } finally {
      setAuthLoading(false);
    }
  };

  const handleLogin = async () => {
    if (!authUsername.trim() || !authPassword.trim()) {
      Alert.alert('Error', 'Enter username and password');
      return;
    }

    try {
      setAuthLoading(true);
      const response = await fetch(`${BACKEND_URL}/login`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'ngrok-skip-browser-warning': 'true',
        },
        body: JSON.stringify({
          username: authUsername.trim(),
          password: authPassword,
        }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || 'Login failed');
      }

      // Save token and go to mode screen
      await AsyncStorage.setItem('authToken', data.token);
      await AsyncStorage.setItem('username', data.username);
      setAuthToken(data.token);
      setLoggedInUser(data.username);
      setMyName(data.username);
      setAuthUsername('');
      setAuthPassword('');
      setScreen('mode');
    } catch (err) {
      Alert.alert('Error', err.message);
    } finally {
      setAuthLoading(false);
    }
  };

  const handleLogout = async () => {
    await AsyncStorage.removeItem('authToken');
    await AsyncStorage.removeItem('username');
    setAuthToken(null);
    setLoggedInUser('');
    setMyName('');
    setScreen('auth');
  };

  // ==================== THRESHOLD CONTROLS ====================
  const adjustThreshold = async (delta) => {
    const newThreshold = Math.max(MIN_THRESHOLD_DB, Math.min(MAX_THRESHOLD_DB, silenceThreshold + delta));
    setSilenceThreshold(newThreshold);
    await AsyncStorage.setItem('silenceThreshold', newThreshold.toString());
  };

  const calibrateThreshold = async () => {
    // Set threshold slightly above current ambient noise level
    const newThreshold = Math.max(MIN_THRESHOLD_DB, Math.min(MAX_THRESHOLD_DB, Math.round(currentDb) + 8));
    setSilenceThreshold(newThreshold);
    await AsyncStorage.setItem('silenceThreshold', newThreshold.toString());
    setStatus(`Calibrated to ${newThreshold} dB`);
  };

  // ==================== VAD LOGIC (expo-av metering) ====================
  const handleMeteringUpdate = useCallback((status) => {
    if (!status.isRecording) return;

    // expo-av provides metering in dB (typically -160 to 0)
    const db = status.metering ?? -60;
    setCurrentDb(db);

    const isSilent = db < silenceThreshold;
    const now = Date.now();

    if (!isSilent) {
      silenceStartTimeRef.current = null;
      if (!speechStartTimeRef.current) {
        speechStartTimeRef.current = now;
        setIsSpeaking(true);
        setStatus('Recording speech...');
      }
    } else {
      if (speechStartTimeRef.current && !silenceStartTimeRef.current) {
        silenceStartTimeRef.current = now;
      }

      if (speechStartTimeRef.current && silenceStartTimeRef.current) {
        const silenceDuration = now - silenceStartTimeRef.current;
        const speechDuration = silenceStartTimeRef.current - speechStartTimeRef.current;

        if (silenceDuration >= SILENCE_DURATION_MS && speechDuration >= MIN_SPEECH_DURATION_MS) {
          setIsSpeaking(false);
          speechStartTimeRef.current = null;
          silenceStartTimeRef.current = null;
          handleSegmentComplete();
        }
      }
    }
  }, [screen, isListening, silenceThreshold]);

  // ==================== SEGMENT HANDLING ====================
  const handleSegmentComplete = async () => {
    if (isProcessing || !recordingRef.current) return;
    setIsProcessing(true);
    setStatus('Processing...');

    try {
      // Stop and get the recording
      await recordingRef.current.stopAndUnloadAsync();
      const uri = recordingRef.current.getURI();
      isRecordingRef.current = false;
      recordingRef.current = null;

      if (uri) {
        const base64Audio = await FileSystem.readAsStringAsync(uri, {
          encoding: FileSystem.EncodingType.Base64,
        });
        await FileSystem.deleteAsync(uri, { idempotent: true });

        if (screen === 'room') {
          // Room mode - send via WebSocket
          sendWsMessage({ type: 'audio', data: base64Audio });
        } else {
          // Solo mode - send via REST
          await sendToBackendSolo(base64Audio);
        }
      }

      // Restart listening if still active
      if (isListeningRef.current && !isPlayingRef.current) {
        await startListeningInternal();
      }
    } catch (err) {
      console.error('Segment error:', err);
      setStatus('Error: ' + err.message);
    } finally {
      setIsProcessing(false);
    }
  };

  // ==================== SOLO MODE BACKEND ====================
  const sendToBackendSolo = async (base64Audio) => {
    try {
      const response = await fetch(`${BACKEND_URL}/translate`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'ngrok-skip-browser-warning': 'true',
          'Authorization': `Bearer ${authToken}`,
        },
        body: JSON.stringify({ audio: base64Audio }),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `Server error: ${response.status}`);
      }

      const data = await response.json();
      const timestamp = Date.now();

      setMessages(prev => [...prev,
        {
          id: `orig-${timestamp}`,
          type: 'original',
          lang: data.detected_lang,
          text: data.original_text,
          timestamp,
        },
        {
          id: `trans-${timestamp}`,
          type: 'translation',
          lang: data.target_lang,
          text: data.translated_text,
          timestamp,
        },
      ]);

      const audioPath = FileSystem.cacheDirectory + `audio_${timestamp}.wav`;
      await FileSystem.writeAsStringAsync(audioPath, data.audio, {
        encoding: FileSystem.EncodingType.Base64,
      });

      audioQueueRef.current.push(audioPath);
      playNextInQueue();

    } catch (err) {
      console.error('Backend error:', err);
      setStatus('Error: ' + err.message);
    }
  };

  // ==================== AUDIO PLAYBACK ====================
  const playNextInQueue = async () => {
    if (isPlayingRef.current || audioQueueRef.current.length === 0) return;

    isPlayingRef.current = true;
    setIsPlaying(true);
    setStatus('Playing...');

    // In solo mode, stop recording during playback to prevent feedback
    const shouldStopRecording = screen !== 'room';
    if (shouldStopRecording && recordingRef.current) {
      try {
        await recordingRef.current.stopAndUnloadAsync();
        isRecordingRef.current = false;
        recordingRef.current = null;
      } catch (e) {
        console.log('Could not stop recording:', e);
      }
    }

    try {
      await Audio.setAudioModeAsync({
        allowsRecordingIOS: !shouldStopRecording,
        playsInSilentModeIOS: true,
        staysActiveInBackground: false,
        playThroughEarpieceAndroid: false,
      });

      const audioData = audioQueueRef.current.shift();
      let audioPath = audioData;

      // If it's base64 (from room mode), save to file first
      if (!audioData.startsWith('file://') && !audioData.startsWith('/')) {
        audioPath = FileSystem.cacheDirectory + `play_${Date.now()}.wav`;
        await FileSystem.writeAsStringAsync(audioPath, audioData, {
          encoding: FileSystem.EncodingType.Base64,
        });
      }

      if (soundRef.current) await soundRef.current.unloadAsync();

      const { sound } = await Audio.Sound.createAsync(
        { uri: audioPath },
        { shouldPlay: true, volume: 1.0 }
      );
      soundRef.current = sound;

      sound.setOnPlaybackStatusUpdate(async (playbackStatus) => {
        if (playbackStatus.didJustFinish) {
          await FileSystem.deleteAsync(audioPath, { idempotent: true });
          isPlayingRef.current = false;
          setIsPlaying(false);

          if (audioQueueRef.current.length > 0) {
            playNextInQueue();
          } else {
            await Audio.setAudioModeAsync({
              allowsRecordingIOS: true,
              playsInSilentModeIOS: true,
              staysActiveInBackground: false,
            });
            // Restart recording in solo mode after playback
            if (isListeningRef.current && shouldStopRecording) {
              await startListeningInternal();
            }
            setStatus(screen === 'room' ? 'Connected' : 'Listening...');
          }
        }
      });
    } catch (err) {
      console.error('Playback error:', err);
      isPlayingRef.current = false;
      setIsPlaying(false);
    }
  };

  // ==================== RECORDING CONTROL (expo-av) ====================
  const startListeningInternal = async () => {
    try {
      speechStartTimeRef.current = null;
      silenceStartTimeRef.current = null;

      // Stop any existing recording
      if (recordingRef.current) {
        try {
          await recordingRef.current.stopAndUnloadAsync();
        } catch (e) {}
        recordingRef.current = null;
      }

      await Audio.setAudioModeAsync({
        allowsRecordingIOS: true,
        playsInSilentModeIOS: true,
        staysActiveInBackground: false,
      });

      // Create new recording with metering enabled
      const recording = new Audio.Recording();
      await recording.prepareToRecordAsync({
        android: {
          extension: '.wav',
          outputFormat: Audio.AndroidOutputFormat.DEFAULT,
          audioEncoder: Audio.AndroidAudioEncoder.DEFAULT,
          sampleRate: 16000,
          numberOfChannels: 1,
          bitRate: 256000,
        },
        ios: {
          extension: '.wav',
          audioQuality: Audio.IOSAudioQuality.HIGH,
          sampleRate: 16000,
          numberOfChannels: 1,
          bitRate: 256000,
          linearPCMBitDepth: 16,
          linearPCMIsBigEndian: false,
          linearPCMIsFloat: false,
        },
        web: {},
        isMeteringEnabled: true,
      });

      // Set up metering callback for VAD
      recording.setOnRecordingStatusUpdate(handleMeteringUpdate);
      recording.setProgressUpdateInterval(100); // Update every 100ms

      await recording.startAsync();
      recordingRef.current = recording;
      isRecordingRef.current = true;

    } catch (err) {
      console.error('Start listening error:', err);
      setStatus('Error: ' + err.message);
    }
  };

  const stopListeningInternal = async () => {
    if (recordingRef.current) {
      try {
        await recordingRef.current.stopAndUnloadAsync();
      } catch (e) {}
      recordingRef.current = null;
      isRecordingRef.current = false;
    }
  };

  const toggleListening = async () => {
    if (isListening) {
      setIsListening(false);
      isListeningRef.current = false;
      await stopListeningInternal();
      setStatus(screen === 'room' ? 'Paused' : 'Stopped');
    } else {
      setIsListening(true);
      isListeningRef.current = true;
      await startListeningInternal();
      setStatus('Listening...');
    }
  };

  // ==================== UI HELPERS ====================
  const getLangFlag = (lang) => {
    if (lang === 'en' || lang === 'english') return '🇬🇧';
    if (lang === 'th' || lang === 'thai') return '🇹🇭';
    return '🌐';
  };

  const getDeepLinkUrl = () => {
    return Linking.createURL(`join/${roomCode}`);
  };

  const getDbBarWidth = () => {
    const normalized = Math.max(0, Math.min(100, ((currentDb + 60) / 60) * 100));
    return `${normalized}%`;
  };

  const getThresholdPosition = () => {
    const normalized = Math.max(0, Math.min(100, ((silenceThreshold + 60) / 60) * 100));
    return `${normalized}%`;
  };

  // ==================== RENDER FUNCTIONS ====================

  // Loading Screen
  const renderLoading = () => (
    <View style={styles.container}>
      <StatusBar style="light" />
      <View style={styles.centeredContent}>
        <ActivityIndicator size="large" color="#3498db" />
        <Text style={styles.loadingText}>Loading...</Text>
      </View>
    </View>
  );

  // Auth Choice Screen
  const renderAuth = () => (
    <View style={styles.container}>
      <StatusBar style="light" />
      <View style={styles.header}>
        <Text style={styles.title}>Thai-English</Text>
        <Text style={styles.subtitle}>Translator V3</Text>
      </View>

      <View style={styles.modeContainer}>
        <Pressable
          style={({ pressed }) => [styles.modeButton, pressed && styles.modeButtonPressed]}
          onPress={() => setScreen('login')}
        >
          <Text style={styles.modeIcon}>🔑</Text>
          <Text style={styles.modeTitle}>LOGIN</Text>
          <Text style={styles.modeDesc}>I have an account</Text>
        </Pressable>

        <Pressable
          style={({ pressed }) => [styles.modeButton, styles.modeButtonRoom, pressed && styles.modeButtonPressed]}
          onPress={() => setScreen('register')}
        >
          <Text style={styles.modeIcon}>✨</Text>
          <Text style={styles.modeTitle}>REGISTER</Text>
          <Text style={styles.modeDesc}>Create new account</Text>
        </Pressable>
      </View>
    </View>
  );

  // Login Screen
  const renderLogin = () => (
    <KeyboardAvoidingView
      style={styles.container}
      behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
    >
      <StatusBar style="light" />
      <View style={styles.header}>
        <Text style={styles.title}>Login</Text>
      </View>

      <View style={styles.formContainer}>
        <Text style={styles.label}>Username:</Text>
        <TextInput
          style={styles.input}
          value={authUsername}
          onChangeText={setAuthUsername}
          placeholder="Your username"
          placeholderTextColor="#666"
          autoCapitalize="none"
          autoCorrect={false}
        />

        <Text style={styles.label}>Password:</Text>
        <TextInput
          style={styles.input}
          value={authPassword}
          onChangeText={setAuthPassword}
          placeholder="Your password"
          placeholderTextColor="#666"
          secureTextEntry
        />

        <Pressable
          style={({ pressed }) => [styles.actionButton, pressed && styles.actionButtonPressed]}
          onPress={handleLogin}
          disabled={authLoading}
        >
          {authLoading ? (
            <ActivityIndicator color="#fff" />
          ) : (
            <Text style={styles.actionButtonText}>LOGIN</Text>
          )}
        </Pressable>
      </View>

      <Pressable style={styles.backButton} onPress={() => { setScreen('auth'); setAuthUsername(''); setAuthPassword(''); }}>
        <Text style={styles.backButtonText}>← Back</Text>
      </Pressable>
    </KeyboardAvoidingView>
  );

  // Register Screen
  const renderRegister = () => (
    <KeyboardAvoidingView
      style={styles.container}
      behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
    >
      <StatusBar style="light" />
      <View style={styles.header}>
        <Text style={styles.title}>Register</Text>
      </View>

      <View style={styles.formContainer}>
        <Text style={styles.label}>Username:</Text>
        <TextInput
          style={styles.input}
          value={authUsername}
          onChangeText={setAuthUsername}
          placeholder="Choose a username"
          placeholderTextColor="#666"
          autoCapitalize="none"
          autoCorrect={false}
        />

        <Text style={styles.label}>Password:</Text>
        <TextInput
          style={styles.input}
          value={authPassword}
          onChangeText={setAuthPassword}
          placeholder="Choose a password"
          placeholderTextColor="#666"
          secureTextEntry
        />

        <Pressable
          style={({ pressed }) => [styles.actionButton, styles.actionButtonSecondary, pressed && styles.actionButtonPressed]}
          onPress={handleRegister}
          disabled={authLoading}
        >
          {authLoading ? (
            <ActivityIndicator color="#fff" />
          ) : (
            <Text style={styles.actionButtonText}>REGISTER</Text>
          )}
        </Pressable>
      </View>

      <Pressable style={styles.backButton} onPress={() => { setScreen('auth'); setAuthUsername(''); setAuthPassword(''); }}>
        <Text style={styles.backButtonText}>← Back</Text>
      </Pressable>
    </KeyboardAvoidingView>
  );

  // Mode Selection Screen
  const renderModeSelector = () => (
    <View style={styles.container}>
      <StatusBar style="light" />
      <View style={styles.header}>
        <Text style={styles.title}>Thai-English</Text>
        <Text style={styles.subtitle}>Translator V3</Text>
        <Text style={styles.userInfo}>Logged in as: {loggedInUser}</Text>
      </View>

      <View style={styles.modeContainer}>
        <Pressable
          style={({ pressed }) => [styles.modeButton, pressed && styles.modeButtonPressed]}
          onPress={() => { setScreen('solo'); setStatus(''); }}
        >
          <Text style={styles.modeIcon}>👤</Text>
          <Text style={styles.modeTitle}>SOLO MODE</Text>
          <Text style={styles.modeDesc}>Translate by yourself</Text>
        </Pressable>

        <Pressable
          style={({ pressed }) => [styles.modeButton, styles.modeButtonRoom, pressed && styles.modeButtonPressed]}
          onPress={() => setScreen('roomSetup')}
        >
          <Text style={styles.modeIcon}>👥</Text>
          <Text style={styles.modeTitle}>ROOM MODE</Text>
          <Text style={styles.modeDesc}>Two-person conversation</Text>
        </Pressable>
      </View>

      <Pressable style={styles.logoutButton} onPress={handleLogout}>
        <Text style={styles.logoutButtonText}>Logout</Text>
      </Pressable>
    </View>
  );

  // Room Setup Screen
  const renderRoomSetup = () => (
    <KeyboardAvoidingView
      style={styles.container}
      behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
    >
      <StatusBar style="light" />
      <View style={styles.header}>
        <Text style={styles.title}>Room Mode</Text>
      </View>

      <View style={styles.formContainer}>
        <Text style={styles.label}>Your name:</Text>
        <TextInput
          style={styles.input}
          value={myName}
          onChangeText={setMyName}
          placeholder="e.g. John"
          placeholderTextColor="#666"
          maxLength={20}
        />

        <Pressable
          style={({ pressed }) => [styles.actionButton, pressed && styles.actionButtonPressed]}
          onPress={createRoom}
        >
          <Text style={styles.actionButtonText}>CREATE ROOM</Text>
        </Pressable>

        <View style={styles.divider}>
          <View style={styles.dividerLine} />
          <Text style={styles.dividerText}>or</Text>
          <View style={styles.dividerLine} />
        </View>

        <Text style={styles.label}>Room code:</Text>
        <TextInput
          style={styles.input}
          value={roomCode}
          onChangeText={setRoomCode}
          placeholder="e.g. 1234"
          placeholderTextColor="#666"
          keyboardType="number-pad"
          maxLength={4}
        />

        <Pressable
          style={({ pressed }) => [styles.actionButton, styles.actionButtonSecondary, pressed && styles.actionButtonPressed]}
          onPress={() => joinRoom()}
        >
          <Text style={styles.actionButtonText}>JOIN ROOM</Text>
        </Pressable>
      </View>

      <Pressable style={styles.backButton} onPress={() => { setScreen('mode'); setStatus(''); }}>
        <Text style={styles.backButtonText}>← Back</Text>
      </Pressable>
    </KeyboardAvoidingView>
  );

  // Waiting for Partner Screen
  const renderWaiting = () => (
    <View style={styles.container}>
      <StatusBar style="light" />
      <View style={styles.header}>
        <Text style={styles.title}>Waiting for partner</Text>
      </View>

      <View style={styles.qrContainer}>
        <View style={styles.qrBox}>
          <QRCode
            value={getDeepLinkUrl()}
            size={200}
            backgroundColor="#fff"
            color="#000"
          />
        </View>
        <Text style={styles.qrHint}>Partner scans this code</Text>

        <View style={styles.codeBox}>
          <Text style={styles.codeLabel}>Room code:</Text>
          <Text style={styles.codeValue}>{roomCode}</Text>
        </View>

        <ActivityIndicator size="large" color="#3498db" style={{ marginTop: 30 }} />
        <Text style={styles.waitingText}>{status}</Text>
      </View>

      <Pressable style={styles.cancelButton} onPress={leaveRoom}>
        <Text style={styles.cancelButtonText}>Cancel</Text>
      </Pressable>
    </View>
  );

  // Join Room Screen (from deep link)
  const renderJoin = () => (
    <KeyboardAvoidingView
      style={styles.container}
      behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
    >
      <StatusBar style="light" />
      <View style={styles.header}>
        <Text style={styles.title}>Join room</Text>
        <Text style={styles.subtitle}>Code: {pendingRoomCode}</Text>
      </View>

      <View style={styles.formContainer}>
        <Text style={styles.label}>Your name:</Text>
        <TextInput
          style={styles.input}
          value={myName}
          onChangeText={setMyName}
          placeholder="e.g. Somchai"
          placeholderTextColor="#666"
          maxLength={20}
          autoFocus
        />

        <Pressable
          style={({ pressed }) => [styles.actionButton, pressed && styles.actionButtonPressed]}
          onPress={() => joinRoom(pendingRoomCode)}
        >
          <Text style={styles.actionButtonText}>JOIN</Text>
        </Pressable>
      </View>

      <Pressable style={styles.backButton} onPress={() => { setPendingRoomCode(null); setScreen('mode'); setStatus(''); }}>
        <Text style={styles.backButtonText}>← Cancel</Text>
      </Pressable>
    </KeyboardAvoidingView>
  );

  // Solo Conversation Screen
  const renderSoloConversation = () => (
    <View style={styles.container}>
      <StatusBar style="light" />
      <View style={styles.header}>
        <Text style={styles.title}>Solo Conversation</Text>
        <Text style={styles.subtitle}>EN ↔ TH Auto-translate</Text>
      </View>

      <View style={styles.messagesContainer}>
        {messages.length === 0 ? (
          <View style={styles.emptyState}>
            <Text style={styles.emptyIcon}>💬</Text>
            <Text style={styles.emptyText}>Press START and begin speaking</Text>
          </View>
        ) : (
          <FlatList
            ref={flatListRef}
            data={messages}
            renderItem={({ item }) => (
              <View style={[styles.messageBox, item.type === 'translation' && styles.translationBox]}>
                <Text style={styles.messageFlag}>{getLangFlag(item.lang)}</Text>
                <Text style={[styles.messageText, item.type === 'translation' && styles.translationText]}>
                  {item.text}
                </Text>
              </View>
            )}
            keyExtractor={(item) => item.id}
            contentContainerStyle={styles.messagesList}
            onContentSizeChange={() => flatListRef.current?.scrollToEnd()}
          />
        )}
      </View>

      {renderStatusBar()}
      {renderControlButton()}

      <Pressable style={styles.backButton} onPress={() => { setScreen('mode'); setMessages([]); setIsListening(false); isListeningRef.current = false; setStatus(''); }}>
        <Text style={styles.backButtonText}>← Back</Text>
      </Pressable>
    </View>
  );

  // Room Conversation Screen
  const renderRoomConversation = () => (
    <View style={styles.container}>
      <StatusBar style="light" />
      <View style={styles.header}>
        <Text style={styles.title}>👥 {partnerName}</Text>
        <Text style={styles.subtitle}>Room: {roomCode}</Text>
      </View>

      <View style={styles.messagesContainer}>
        {messages.length === 0 ? (
          <View style={styles.emptyState}>
            <Text style={styles.emptyIcon}>💬</Text>
            <Text style={styles.emptyText}>Press START and begin speaking</Text>
          </View>
        ) : (
          <FlatList
            ref={flatListRef}
            data={messages}
            renderItem={({ item }) => (
              <View style={[
                styles.roomMessageBox,
                item.isMe ? styles.roomMessageMine : styles.roomMessageTheirs
              ]}>
                <Text style={styles.roomMessageFrom}>
                  {getLangFlag(item.langFrom)} {item.from}
                </Text>
                <Text style={styles.roomMessageOriginal}>{item.original}</Text>
                <Text style={styles.roomMessageTranslated}>
                  {getLangFlag(item.langTo)} {item.translated}
                </Text>
              </View>
            )}
            keyExtractor={(item) => item.id}
            contentContainerStyle={styles.messagesList}
            onContentSizeChange={() => flatListRef.current?.scrollToEnd()}
          />
        )}
      </View>

      {renderStatusBar()}
      {renderControlButton()}

      <Pressable style={styles.leaveButton} onPress={leaveRoom}>
        <Text style={styles.leaveButtonText}>Leave room</Text>
      </Pressable>
    </View>
  );

  // Shared Status Bar
  const renderStatusBar = () => (
    <View style={styles.statusContainer}>
      {isListening && (
        <>
          <View style={styles.dbMeter}>
            <View style={styles.dbBarContainer}>
              <View style={[
                styles.dbBar,
                {
                  width: getDbBarWidth(),
                  backgroundColor: currentDb > silenceThreshold ? '#27ae60' : '#7f8c8d'
                }
              ]} />
              {/* Threshold marker */}
              <View style={[styles.thresholdMarker, { left: getThresholdPosition() }]} />
            </View>
            <Text style={styles.dbText}>{Math.round(currentDb)} dB</Text>
          </View>
          {/* Threshold controls */}
          <View style={styles.thresholdControls}>
            <Text style={styles.thresholdLabel}>Noise gate:</Text>
            <Pressable
              style={[styles.thresholdButton, silenceThreshold <= MIN_THRESHOLD_DB && styles.thresholdButtonDisabled]}
              onPress={() => adjustThreshold(-5)}
              disabled={silenceThreshold <= MIN_THRESHOLD_DB}
            >
              <Text style={styles.thresholdButtonText}>−</Text>
            </Pressable>
            <Text style={styles.thresholdValue}>{silenceThreshold} dB</Text>
            <Pressable
              style={[styles.thresholdButton, silenceThreshold >= MAX_THRESHOLD_DB && styles.thresholdButtonDisabled]}
              onPress={() => adjustThreshold(5)}
              disabled={silenceThreshold >= MAX_THRESHOLD_DB}
            >
              <Text style={styles.thresholdButtonText}>+</Text>
            </Pressable>
            <Pressable style={styles.calibrateButton} onPress={calibrateThreshold}>
              <Text style={styles.calibrateButtonText}>CAL</Text>
            </Pressable>
          </View>
        </>
      )}
      <View style={styles.statusTextContainer}>
        {isProcessing && <ActivityIndicator size="small" color="#3498db" />}
        <Text style={[
          styles.statusText,
          isSpeaking && styles.statusSpeaking,
          isPlaying && styles.statusPlaying,
        ]}>
          {isSpeaking && '🔴 '}
          {isPlaying && '🔊 '}
          {status}
        </Text>
      </View>
    </View>
  );

  // Shared Control Button
  const renderControlButton = () => (
    <View style={styles.buttonContainer}>
      <Pressable
        onPress={toggleListening}
        disabled={isProcessing}
        style={({ pressed }) => [
          styles.button,
          isListening && styles.buttonActive,
          pressed && styles.buttonPressed,
        ]}
      >
        <Text style={styles.buttonIcon}>{isListening ? '⏹️' : '▶️'}</Text>
        <Text style={styles.buttonText}>{isListening ? 'STOP' : 'START'}</Text>
      </Pressable>
    </View>
  );

  // ==================== MAIN RENDER ====================
  switch (screen) {
    case 'loading': return renderLoading();
    case 'auth': return renderAuth();
    case 'login': return renderLogin();
    case 'register': return renderRegister();
    case 'mode': return renderModeSelector();
    case 'solo': return renderSoloConversation();
    case 'roomSetup': return renderRoomSetup();
    case 'waiting': return renderWaiting();
    case 'join': return renderJoin();
    case 'room': return renderRoomConversation();
    default: return renderLoading();
  }
}

// ==================== STYLES ====================
const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#0f0f1a',
    paddingTop: 60,
  },
  centeredContent: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  loadingText: {
    color: '#888',
    fontSize: 16,
    marginTop: 15,
  },
  header: {
    alignItems: 'center',
    paddingHorizontal: 20,
    marginBottom: 20,
  },
  title: {
    fontSize: 28,
    fontWeight: 'bold',
    color: '#fff',
  },
  subtitle: {
    fontSize: 16,
    color: '#888',
    marginTop: 2,
  },
  userInfo: {
    fontSize: 14,
    color: '#3498db',
    marginTop: 8,
  },

  // Mode Selector
  modeContainer: {
    flex: 1,
    justifyContent: 'center',
    paddingHorizontal: 30,
    gap: 20,
  },
  modeButton: {
    backgroundColor: '#1a1a2e',
    borderRadius: 20,
    padding: 30,
    alignItems: 'center',
    borderWidth: 2,
    borderColor: '#3498db',
  },
  modeButtonRoom: {
    borderColor: '#27ae60',
  },
  modeButtonPressed: {
    opacity: 0.8,
    transform: [{ scale: 0.98 }],
  },
  modeIcon: {
    fontSize: 50,
    marginBottom: 10,
  },
  modeTitle: {
    fontSize: 22,
    fontWeight: 'bold',
    color: '#fff',
  },
  modeDesc: {
    fontSize: 14,
    color: '#888',
    marginTop: 5,
  },

  // Form
  formContainer: {
    flex: 1,
    paddingHorizontal: 30,
    paddingTop: 20,
  },
  label: {
    color: '#888',
    fontSize: 14,
    marginBottom: 8,
    marginTop: 15,
  },
  input: {
    backgroundColor: '#1a1a2e',
    borderRadius: 12,
    padding: 15,
    color: '#fff',
    fontSize: 18,
    borderWidth: 1,
    borderColor: '#333',
  },
  actionButton: {
    backgroundColor: '#3498db',
    borderRadius: 12,
    padding: 18,
    alignItems: 'center',
    marginTop: 20,
  },
  actionButtonSecondary: {
    backgroundColor: '#27ae60',
  },
  actionButtonPressed: {
    opacity: 0.8,
  },
  actionButtonText: {
    color: '#fff',
    fontSize: 18,
    fontWeight: 'bold',
  },
  divider: {
    flexDirection: 'row',
    alignItems: 'center',
    marginVertical: 25,
  },
  dividerLine: {
    flex: 1,
    height: 1,
    backgroundColor: '#333',
  },
  dividerText: {
    color: '#666',
    marginHorizontal: 15,
  },

  // QR / Waiting
  qrContainer: {
    flex: 1,
    alignItems: 'center',
    paddingTop: 20,
  },
  qrBox: {
    backgroundColor: '#fff',
    padding: 20,
    borderRadius: 20,
  },
  qrHint: {
    color: '#888',
    marginTop: 15,
    fontSize: 14,
  },
  codeBox: {
    marginTop: 30,
    alignItems: 'center',
  },
  codeLabel: {
    color: '#888',
    fontSize: 14,
  },
  codeValue: {
    color: '#fff',
    fontSize: 48,
    fontWeight: 'bold',
    letterSpacing: 10,
    marginTop: 5,
  },
  waitingText: {
    color: '#888',
    marginTop: 15,
  },

  // Messages
  messagesContainer: {
    flex: 1,
    paddingHorizontal: 15,
  },
  emptyState: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  emptyIcon: {
    fontSize: 60,
    marginBottom: 20,
  },
  emptyText: {
    color: '#666',
    fontSize: 16,
    textAlign: 'center',
  },
  messagesList: {
    paddingVertical: 10,
  },
  messageBox: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    backgroundColor: '#1a1a2e',
    borderRadius: 12,
    padding: 12,
    marginBottom: 8,
  },
  translationBox: {
    backgroundColor: '#2a2a4a',
    borderLeftWidth: 3,
    borderLeftColor: '#3498db',
    marginBottom: 16,
  },
  messageFlag: {
    fontSize: 20,
    marginRight: 10,
  },
  messageText: {
    flex: 1,
    color: '#ccc',
    fontSize: 16,
    lineHeight: 22,
  },
  translationText: {
    color: '#fff',
    fontWeight: '500',
  },

  // Room Messages
  roomMessageBox: {
    backgroundColor: '#1a1a2e',
    borderRadius: 12,
    padding: 12,
    marginBottom: 12,
    maxWidth: '85%',
  },
  roomMessageMine: {
    alignSelf: 'flex-end',
    backgroundColor: '#1a3a5c',
    borderBottomRightRadius: 4,
  },
  roomMessageTheirs: {
    alignSelf: 'flex-start',
    backgroundColor: '#2a2a4a',
    borderBottomLeftRadius: 4,
  },
  roomMessageFrom: {
    color: '#888',
    fontSize: 12,
    marginBottom: 4,
  },
  roomMessageOriginal: {
    color: '#fff',
    fontSize: 16,
    marginBottom: 8,
  },
  roomMessageTranslated: {
    color: '#3498db',
    fontSize: 14,
    fontStyle: 'italic',
  },

  // Status
  statusContainer: {
    paddingHorizontal: 20,
    paddingVertical: 15,
    borderTopWidth: 1,
    borderTopColor: '#1a1a2e',
  },
  dbMeter: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 10,
  },
  dbBarContainer: {
    flex: 1,
    height: 8,
    backgroundColor: '#1a1a2e',
    borderRadius: 4,
    overflow: 'visible',
    marginRight: 10,
    position: 'relative',
  },
  dbBar: {
    height: '100%',
    borderRadius: 4,
  },
  thresholdMarker: {
    position: 'absolute',
    top: -4,
    width: 3,
    height: 16,
    backgroundColor: '#e74c3c',
    borderRadius: 1,
    marginLeft: -1,
  },
  dbText: {
    color: '#666',
    fontSize: 12,
    width: 50,
    textAlign: 'right',
  },
  thresholdControls: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    marginTop: 8,
    marginBottom: 5,
  },
  thresholdLabel: {
    color: '#666',
    fontSize: 12,
    marginRight: 10,
  },
  thresholdButton: {
    width: 32,
    height: 32,
    borderRadius: 16,
    backgroundColor: '#2a2a4a',
    alignItems: 'center',
    justifyContent: 'center',
    marginHorizontal: 5,
  },
  thresholdButtonDisabled: {
    opacity: 0.3,
  },
  thresholdButtonText: {
    color: '#fff',
    fontSize: 18,
    fontWeight: 'bold',
  },
  thresholdValue: {
    color: '#fff',
    fontSize: 14,
    fontWeight: 'bold',
    minWidth: 55,
    textAlign: 'center',
  },
  calibrateButton: {
    paddingHorizontal: 12,
    paddingVertical: 6,
    backgroundColor: '#3498db',
    borderRadius: 12,
    marginLeft: 10,
  },
  calibrateButtonText: {
    color: '#fff',
    fontSize: 12,
    fontWeight: 'bold',
  },
  statusTextContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
  },
  statusText: {
    color: '#888',
    fontSize: 14,
    marginLeft: 8,
  },
  statusSpeaking: {
    color: '#e74c3c',
  },
  statusPlaying: {
    color: '#27ae60',
  },

  // Control Button
  buttonContainer: {
    alignItems: 'center',
    paddingVertical: 15,
  },
  button: {
    width: 120,
    height: 120,
    borderRadius: 60,
    backgroundColor: '#27ae60',
    alignItems: 'center',
    justifyContent: 'center',
    shadowColor: '#27ae60',
    shadowOffset: { width: 0, height: 0 },
    shadowOpacity: 0.5,
    shadowRadius: 15,
    elevation: 10,
  },
  buttonActive: {
    backgroundColor: '#e74c3c',
    shadowColor: '#e74c3c',
  },
  buttonPressed: {
    transform: [{ scale: 0.95 }],
    opacity: 0.9,
  },
  buttonIcon: {
    fontSize: 35,
  },
  buttonText: {
    color: '#fff',
    fontSize: 14,
    fontWeight: 'bold',
    marginTop: 5,
  },

  // Navigation Buttons
  backButton: {
    alignItems: 'center',
    paddingVertical: 15,
    marginBottom: 20,
  },
  backButtonText: {
    color: '#666',
    fontSize: 16,
  },
  cancelButton: {
    alignItems: 'center',
    paddingVertical: 15,
    marginBottom: 30,
  },
  cancelButtonText: {
    color: '#e74c3c',
    fontSize: 16,
  },
  leaveButton: {
    alignItems: 'center',
    paddingVertical: 15,
    marginBottom: 20,
  },
  leaveButtonText: {
    color: '#e74c3c',
    fontSize: 14,
  },
  logoutButton: {
    alignItems: 'center',
    paddingVertical: 15,
    marginBottom: 30,
  },
  logoutButtonText: {
    color: '#888',
    fontSize: 14,
  },
});
