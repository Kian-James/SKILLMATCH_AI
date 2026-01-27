import 'package:flutter/material.dart';
import '../models/resume_model.dart';
import '../widgets/skill_chip.dart';
import 'job_match_screen.dart';
import 'skill_gap_screen.dart';

class ResumeResultScreen extends StatelessWidget {
  final ResumeModel resume;

  const ResumeResultScreen({Key? key, required this.resume}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Resume Skills')),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text(
              'Extracted Skills:',
              style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 10),

            // Display skills as chips
            Wrap(
              spacing: 8,
              runSpacing: 4,
              children: resume.skills
                  .map((skill) => SkillChip(skill: skill))
                  .toList(),
            ),

            const SizedBox(height: 30),

            // Buttons to navigate
            ElevatedButton.icon(
              icon: const Icon(Icons.work),
              label: const Text('Find Matching Jobs'),
              onPressed: () {
                Navigator.push(
                  context,
                  MaterialPageRoute(
                    builder: (context) => JobMatchScreen(userId: resume.userId),
                  ),
                );
              },
            ),
            const SizedBox(height: 10),
            ElevatedButton.icon(
              icon: const Icon(Icons.insights),
              label: const Text('Analyze Skill Gap'),
              onPressed: () {
                Navigator.push(
                  context,
                  MaterialPageRoute(
                    builder: (context) =>
                        SkillGapScreen(userId: resume.userId),
                  ),
                );
              },
            ),
          ],
        ),
      ),
    );
  }
}
