import 'package:flutter/material.dart';
import '../models/job_model.dart';
import 'skill_chip.dart';

class JobCard extends StatelessWidget {
  final JobModel job;

  const JobCard({Key? key, required this.job}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Card(
      elevation: 3,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
      child: Padding(
        padding: const EdgeInsets.all(12.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(job.title, style: const TextStyle(fontSize: 18, fontWeight: FontWeight.bold)),
            const SizedBox(height: 4),
            Text(job.company, style: TextStyle(color: Colors.grey.shade700)),
            const SizedBox(height: 2),
            Text(job.location, style: TextStyle(color: Colors.grey.shade600)),
            const SizedBox(height: 8),
            Wrap(
              spacing: 6,
              runSpacing: 4,
              children: job.skills.map((s) => SkillChip(skill: s)).toList(),
            ),
          ],
        ),
      ),
    );
  }
}
