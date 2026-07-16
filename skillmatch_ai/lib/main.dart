// lib/main.dart

import 'package:firebase_auth/firebase_auth.dart';
import 'package:firebase_core/firebase_core.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import 'firebase_options.dart';
import 'models/models.dart';
import 'screens/home_shell.dart';
import 'screens/login_screen.dart';
import 'screens/splash_screen.dart';
import 'theme.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();

  // Lock orientation on mobile only — not supported on web
  if (!kIsWeb) {
    await SystemChrome.setPreferredOrientations([
      DeviceOrientation.portraitUp,
      DeviceOrientation.portraitDown,
    ]);
  }

  await Firebase.initializeApp(
    options: DefaultFirebaseOptions.currentPlatform,
  );

  runApp(const ResumeAnalyzerApp());
}

class ResumeAnalyzerApp extends StatefulWidget {
  const ResumeAnalyzerApp({super.key});

  static _ResumeAnalyzerAppState of(BuildContext context) =>
      context.findAncestorStateOfType<_ResumeAnalyzerAppState>()!;

  @override
  State<ResumeAnalyzerApp> createState() => _ResumeAnalyzerAppState();
}

class _ResumeAnalyzerAppState extends State<ResumeAnalyzerApp> {
  ThemeMode _themeMode = ThemeMode.light;

  void toggleTheme() {
    setState(() {
      _themeMode =
          _themeMode == ThemeMode.dark ? ThemeMode.light : ThemeMode.dark;
    });
  }

  bool get isDark => _themeMode == ThemeMode.dark;

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'SkillMatch AI',
      debugShowCheckedModeBanner: false,
      theme: AppTheme.light,
      darkTheme: AppTheme.dark,
      themeMode: _themeMode,
      home: const _AuthGate(),
    );
  }
}

class _AuthGate extends StatefulWidget {
  const _AuthGate();

  @override
  State<_AuthGate> createState() => _AuthGateState();
}

class _AuthGateState extends State<_AuthGate> {
  // Enforces a minimum splash duration so the entrance animation
  // (and poster content) always gets to play out, even if Firebase
  // resolves auth state almost instantly.
  bool _minSplashElapsed = false;

  @override
  void initState() {
    super.initState();
    Future.delayed(const Duration(milliseconds: 1600), () {
      if (mounted) setState(() => _minSplashElapsed = true);
    });
  }

  @override
  Widget build(BuildContext context) {
    return StreamBuilder<User?>(
      stream: FirebaseAuth.instance.authStateChanges(),
      builder: (context, snap) {
        final authResolved = snap.connectionState != ConnectionState.waiting;

        if (!authResolved || !_minSplashElapsed) {
          return const SplashScreen();
        }

        final firebaseUser = snap.data;

        if (firebaseUser != null) {
          final user = ResumeUser.fromFirebase(
            uid: firebaseUser.uid,
            displayName: firebaseUser.displayName,
            email: firebaseUser.email,
            photoUrl: firebaseUser.photoURL,
          );
          return HomeShell(user: user);
        }

        return const LoginScreen();
      },
    );
  }
}
