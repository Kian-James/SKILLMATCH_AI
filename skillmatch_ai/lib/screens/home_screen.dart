import 'package:flutter/material.dart';
import '../services/auth_service.dart';

class HomeScreen extends StatelessWidget {
  const HomeScreen({Key? key}) : super(key: key);

  void _logout(BuildContext context) async {
    await AuthService().signOut();
    Navigator.pushReplacementNamed(context, '/login');
  }

  void _goToResumeUpload(BuildContext context) {
    final userId = AuthService().currentUser?.uid ?? '';
    Navigator.pushNamed(context, '/resumeUpload', arguments: {'userId': userId});
  }

  void _goToSkillGap(BuildContext context) {
    final userId = AuthService().currentUser?.uid ?? '';
    Navigator.pushNamed(context, '/skillGap', arguments: {'userId': userId});
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Home'),
        actions: [
          IconButton(
            icon: const Icon(Icons.logout),
            onPressed: () => _logout(context),
          ),
        ],
      ),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          children: [
            ElevatedButton.icon(
              icon: const Icon(Icons.upload_file),
              label: const Text('Upload Resume'),
              onPressed: () => _goToResumeUpload(context),
            ),
            const SizedBox(height: 12),
            ElevatedButton.icon(
              icon: const Icon(Icons.insights),
              label: const Text('Skill Gap Analysis'),
              onPressed: () => _goToSkillGap(context),
            ),
          ],
        ),
      ),
    );
  }
}
