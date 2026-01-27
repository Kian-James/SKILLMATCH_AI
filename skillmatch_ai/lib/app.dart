import 'package:flutter/material.dart';
import 'screens/login_screen.dart';
import 'screens/home_screen.dart';
import 'screens/signup_screen.dart';
import 'screens/resume_upload_screen.dart';
import 'screens/resume_result_screen.dart';
import 'screens/job_match_screen.dart';
import 'screens/skill_gap_screen.dart';

class MyApp extends StatelessWidget {
  const MyApp({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'AI Resume Matcher',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        primarySwatch: Colors.blue,
      ),
      initialRoute: '/login',
      routes: {
        '/login': (context) => const LoginScreen(),
        '/signup': (context) => const SignupScreen(),
        '/home': (context) => const HomeScreen(),
        // Optional: dynamic routing for screens that require arguments
      },
      onGenerateRoute: (settings) {
        // Handles routes with arguments
        switch (settings.name) {
          case '/resumeUpload':
            final args = settings.arguments as Map<String, dynamic>;
            return MaterialPageRoute(
              builder: (_) => ResumeUploadScreen(userId: args['userId']),
            );
          case '/resumeResult':
            final args = settings.arguments as Map<String, dynamic>;
            return MaterialPageRoute(
              builder: (_) => ResumeResultScreen(resume: args['resume']),
            );
          case '/jobMatch':
            final args = settings.arguments as Map<String, dynamic>;
            return MaterialPageRoute(
              builder: (_) => JobMatchScreen(userId: args['userId']),
            );
          case '/skillGap':
            final args = settings.arguments as Map<String, dynamic>;
            return MaterialPageRoute(
              builder: (_) => SkillGapScreen(userId: args['userId']),
            );
          default:
            return null;
        }
      },
    );
  }
}
