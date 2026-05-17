// lib/main.dart

import 'package:firebase_auth/firebase_auth.dart';
import 'package:firebase_core/firebase_core.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import 'models/models.dart';
import 'screens/home_shell.dart';
import 'screens/login_screen.dart';
import 'theme.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();

  await SystemChrome.setPreferredOrientations([
    DeviceOrientation.portraitUp,
    DeviceOrientation.portraitDown,
  ]);

  // Place google-services.json in android/app/
  // Place GoogleService-Info.plist in ios/Runner/
  await Firebase.initializeApp();

  runApp(const ResumeAnalyzerApp());
}

class ResumeAnalyzerApp extends StatelessWidget {
  const ResumeAnalyzerApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Resume Analyzer',
      debugShowCheckedModeBanner: false,
      theme: AppTheme.theme,
      home: const _AuthGate(),
    );
  }
}

/// Listens to Firebase auth state.
/// • Signed-in Firebase user  → HomeShell (using Firebase UID as user ID)
/// • No Firebase session      → LoginScreen
///
/// Guest users are handled entirely inside LoginScreen — they never go through
/// Firebase so this stream never fires for them; LoginScreen pushes directly
/// to HomeShell with a guest ResumeUser.
class _AuthGate extends StatelessWidget {
  const _AuthGate();

  @override
  Widget build(BuildContext context) {
    return StreamBuilder<User?>(
      stream: FirebaseAuth.instance.authStateChanges(),
      builder: (context, snap) {
        // Waiting for Firebase to initialise
        if (snap.connectionState == ConnectionState.waiting) {
          return const Scaffold(
            body: Center(
              child: CircularProgressIndicator(color: AppTheme.teal500),
            ),
          );
        }

        final firebaseUser = snap.data;

        if (firebaseUser != null) {
          // Rebuild ResumeUser from the current Firebase user
          final user = ResumeUser.fromFirebase(
            uid:         firebaseUser.uid,
            displayName: firebaseUser.displayName,
            email:       firebaseUser.email,
            photoUrl:    firebaseUser.photoURL,
          );
          return HomeShell(user: user);
        }

        return const LoginScreen();
      },
    );
  }
}
