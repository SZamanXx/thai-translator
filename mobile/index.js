// Custom entry point to suppress Expo SDK 54 bridgeless warning
const originalConsoleError = console.error;
console.error = (...args) => {
  const message = args[0]?.toString() || '';
  if (
    message.includes('disableEventLoopOnBridgeless') ||
    message.includes('Could not access feature flag')
  ) {
    return; // Suppress this specific error
  }
  originalConsoleError.apply(console, args);
};

// Now load the app
import 'expo/AppEntry';
