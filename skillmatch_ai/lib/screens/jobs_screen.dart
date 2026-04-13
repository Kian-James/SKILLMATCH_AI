// screens/jobs_screen.dart
// Shows the top 5 recommended jobs from the resume analysis.
// Tap any job card to open a detail sheet with all required skills + certs.

import 'package:flutter/material.dart';
import 'package:flutter_animate/flutter_animate.dart';

import '../models/resume_analysis.dart';
import '../widgets/skill_chip.dart';

class JobsScreen extends StatelessWidget {
  final ResumeAnalysis analysis;

  const JobsScreen({super.key, required this.analysis});

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    final jobs = analysis.topJobMatches;

    return Scaffold(
      appBar: AppBar(
        title: const Text('Top Job Matches'),
        backgroundColor: Colors.transparent,
        elevation: 0,
      ),
      body: jobs.isEmpty
          ? const Center(
        child: Text('No job matches found. Try uploading again.'),
      )
          : ListView.builder(
        padding: const EdgeInsets.all(20),
        itemCount: jobs.length,
        itemBuilder: (context, index) {
          final job = jobs[index];
          final rank = index + 1;
          return _JobCard(
            job: job,
            rank: rank,
            userSkills: analysis.skills,
          )
              .animate()
              .fadeIn(
              delay: Duration(milliseconds: 80 * index))
              .slideX(begin: 0.05);
        },
      ),
    );
  }
}

class _JobCard extends StatelessWidget {
  final JobMatch job;
  final int rank;
  final List<String> userSkills;

  const _JobCard({
    required this.job,
    required this.rank,
    required this.userSkills,
  });

  void _showDetail(BuildContext context) {
    showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(24)),
      ),
      builder: (_) => _JobDetailSheet(job: job, userSkills: userSkills),
    );
  }

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    final isTop = rank == 1;

    final scoreColor = switch (job.scoreLevel) {
      ScoreLevel.high => Colors.green,
      ScoreLevel.medium => Colors.orange,
      ScoreLevel.low => Colors.red,
    };

    return GestureDetector(
      onTap: () => _showDetail(context),
      child: Container(
        margin: const EdgeInsets.only(bottom: 16),
        padding: const EdgeInsets.all(18),
        decoration: BoxDecoration(
          color: cs.surfaceContainerLow,
          borderRadius: BorderRadius.circular(18),
          border: isTop ? Border.all(color: cs.primary, width: 2) : null,
          boxShadow: [
            BoxShadow(
              color: cs.shadow.withOpacity(0.06),
              blurRadius: 10,
              offset: const Offset(0, 3),
            ),
          ],
        ),
        child: Row(
          children: [
            // Rank badge
            Container(
              width: 48,
              height: 48,
              decoration: BoxDecoration(
                color: isTop ? cs.primary : cs.primaryContainer,
                borderRadius: BorderRadius.circular(12),
              ),
              child: Center(
                child: Text(
                  '#$rank',
                  style: TextStyle(
                    color: isTop ? cs.onPrimary : cs.primary,
                    fontWeight: FontWeight.bold,
                    fontSize: 15,
                  ),
                ),
              ),
            ),

            const SizedBox(width: 16),

            // Title + skill count
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    job.jobTitle,
                    style: const TextStyle(
                        fontWeight: FontWeight.w700, fontSize: 16),
                  ),
                  const SizedBox(height: 4),
                  Text(
                    '${job.skills.length} required skills',
                    style: TextStyle(
                        color: cs.onSurfaceVariant, fontSize: 13),
                  ),
                ],
              ),
            ),

            // Score badge
            Container(
              padding:
              const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
              decoration: BoxDecoration(
                color: scoreColor.withOpacity(0.12),
                borderRadius: BorderRadius.circular(20),
              ),
              child: Text(
                '${job.matchScore.toStringAsFixed(0)}%',
                style: TextStyle(
                    color: scoreColor,
                    fontWeight: FontWeight.bold,
                    fontSize: 15),
              ),
            ),

            const SizedBox(width: 8),
            Icon(Icons.chevron_right, color: cs.onSurfaceVariant),
          ],
        ),
      ),
    );
  }
}

// ── Job Detail Bottom Sheet ───────────────────────────────────────────────────

class _JobDetailSheet extends StatelessWidget {
  final JobMatch job;
  final List<String> userSkills;

  const _JobDetailSheet({required this.job, required this.userSkills});

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;

    // Use backend-provided matchedSkills (exact + keyword + semantic matching)
    // instead of client-side string comparison. The backend correctly identifies
    // matches like "Version Control Systems" ↔ "Git Version Control" via
    // semantic similarity — a simple toLowerCase().contains() check misses these
    // and causes chips to show red even when the backend scored them as matched.
    final matchedSkillsLower = job.matchedSkills.map((s) => s.toLowerCase()).toSet();

    final scoreColor = switch (job.scoreLevel) {
      ScoreLevel.high => Colors.green,
      ScoreLevel.medium => Colors.orange,
      ScoreLevel.low => Colors.red,
    };

    return DraggableScrollableSheet(
      initialChildSize: 0.7,
      maxChildSize: 0.95,
      minChildSize: 0.4,
      expand: false,
      builder: (_, scrollCtrl) => Container(
        decoration: BoxDecoration(
          color: cs.surface,
          borderRadius:
          const BorderRadius.vertical(top: Radius.circular(24)),
        ),
        child: Column(
          children: [
            // Handle
            Container(
              width: 40,
              height: 4,
              margin: const EdgeInsets.only(top: 12, bottom: 8),
              decoration: BoxDecoration(
                color: cs.outlineVariant,
                borderRadius: BorderRadius.circular(2),
              ),
            ),

            Expanded(
              child: ListView(
                controller: scrollCtrl,
                padding: const EdgeInsets.fromLTRB(24, 8, 24, 32),
                children: [
                  // Job title + score
                  Row(
                    children: [
                      Expanded(
                        child: Text(
                          job.jobTitle,
                          style: Theme.of(context)
                              .textTheme
                              .headlineSmall
                              ?.copyWith(fontWeight: FontWeight.w800),
                        ),
                      ),
                      Container(
                        padding: const EdgeInsets.symmetric(
                            horizontal: 14, vertical: 8),
                        decoration: BoxDecoration(
                          color: scoreColor.withOpacity(0.12),
                          borderRadius: BorderRadius.circular(20),
                        ),
                        child: Text(
                          '${job.matchScore.toStringAsFixed(0)}% match',
                          style: TextStyle(
                              color: scoreColor,
                              fontWeight: FontWeight.bold),
                        ),
                      ),
                    ],
                  ),

                  const SizedBox(height: 24),

                  // Required Skills — colour-coded by whether user has them
                  Text(
                    'Required Skills',
                    style: Theme.of(context)
                        .textTheme
                        .titleMedium
                        ?.copyWith(fontWeight: FontWeight.w700),
                  ),
                  const SizedBox(height: 4),
                  Text(
                    'Green = you have it   ·   Red = you\'re missing it',
                    style: TextStyle(
                        color: cs.onSurfaceVariant, fontSize: 12),
                  ),
                  const SizedBox(height: 12),
                  Wrap(
                    spacing: 8,
                    runSpacing: 8,
                    children: job.skills.map((skill) {
                      final have =
                      matchedSkillsLower.contains(skill.toLowerCase());
                      return SkillChip(
                        skill: skill,
                        type: have
                            ? SkillChipType.present
                            : SkillChipType.missing,
                      );
                    }).toList(),
                  ),

                  // Certifications
                  if (job.certifications.isNotEmpty) ...[
                    const SizedBox(height: 24),
                    Text(
                      'Recommended Certifications',
                      style: Theme.of(context)
                          .textTheme
                          .titleMedium
                          ?.copyWith(fontWeight: FontWeight.w700),
                    ),
                    const SizedBox(height: 12),
                    Wrap(
                      spacing: 8,
                      runSpacing: 8,
                      children: job.certifications
                          .map((c) =>
                          SkillChip(skill: c, type: SkillChipType.cert))
                          .toList(),
                    ),
                  ],

                  const SizedBox(height: 24),

                  // Summary line
                  Container(
                    padding: const EdgeInsets.all(14),
                    decoration: BoxDecoration(
                      color: cs.surfaceContainerLow,
                      borderRadius: BorderRadius.circular(12),
                    ),
                    child: Text(
                      'You have ${job.skills.where((s) => matchedSkillsLower.contains(s.toLowerCase())).length} of ${job.skills.length} required skills for this role.',
                      style: TextStyle(
                          color: cs.onSurfaceVariant, fontSize: 13),
                      textAlign: TextAlign.center,
                    ),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}