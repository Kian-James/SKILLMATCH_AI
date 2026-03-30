// screens/recommendations_screen.dart
// Standalone job recommendation screen.
// User enters skills manually to get recommendations (or results auto-populate from Upload tab).

import 'package:flutter/material.dart';
import 'package:flutter_animate/flutter_animate.dart';
import '../models/resume_analysis.dart';
import '../services/api_service.dart';
import '../widgets/skill_chip.dart';

class RecommendationsScreen extends StatefulWidget {
  const RecommendationsScreen({super.key});

  @override
  State<RecommendationsScreen> createState() => _RecommendationsScreenState();
}

class _RecommendationsScreenState extends State<RecommendationsScreen> {
  final _api = ApiService();
  final _skillsCtrl = TextEditingController();
  final _certsCtrl = TextEditingController();

  bool _isLoading = false;
  String? _error;
  List<JobMatch> _jobs = [];
  int _querySkillCount = 0;

  Future<void> _recommend() async {
    final raw = _skillsCtrl.text.trim();
    if (raw.isEmpty) return;

    final skills = raw.split(',').map((s) => s.trim()).where((s) => s.isNotEmpty).toList();
    final certs = _certsCtrl.text
        .split(',')
        .map((s) => s.trim())
        .where((s) => s.isNotEmpty)
        .toList();

    setState(() {
      _isLoading = true;
      _error = null;
      _jobs = [];
    });

    try {
      final matches = await _api.recommendJobs(skills: skills, certifications: certs);
      setState(() {
        _jobs = matches;
        _querySkillCount = skills.length;
      });
    } catch (e) {
      setState(() => _error = ApiService.extractError(e));
    } finally {
      setState(() => _isLoading = false);
    }
  }

  @override
  void dispose() {
    _skillsCtrl.dispose();
    _certsCtrl.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;

    return SingleChildScrollView(
      padding: const EdgeInsets.all(20),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          // Info banner
          Container(
            padding: const EdgeInsets.all(14),
            decoration: BoxDecoration(
              color: cs.secondaryContainer,
              borderRadius: BorderRadius.circular(12),
            ),
            child: Row(
              children: [
                Icon(Icons.info_outline, color: cs.secondary, size: 18),
                const SizedBox(width: 10),
                Expanded(
                  child: Text(
                    'Enter your skills below, or upload a resume on the Resume tab to auto-populate.',
                    style: TextStyle(color: cs.onSecondaryContainer, fontSize: 13),
                  ),
                ),
              ],
            ),
          ).animate().fadeIn(),

          const SizedBox(height: 20),

          // Input card
          Container(
            padding: const EdgeInsets.all(20),
            decoration: BoxDecoration(
              color: cs.surfaceContainerLow,
              borderRadius: BorderRadius.circular(16),
            ),
            child: Column(
              children: [
                TextField(
                  controller: _skillsCtrl,
                  maxLines: 3,
                  decoration: const InputDecoration(
                    labelText: 'Your Skills *',
                    hintText: 'Python, Docker, Machine Learning, React...',
                    prefixIcon: Icon(Icons.psychology_outlined),
                    alignLabelWithHint: true,
                  ),
                ),
                const SizedBox(height: 14),
                TextField(
                  controller: _certsCtrl,
                  decoration: const InputDecoration(
                    labelText: 'Certifications (optional)',
                    hintText: 'AWS Certified Developer, CKA...',
                    prefixIcon: Icon(Icons.verified_outlined),
                  ),
                ),
                const SizedBox(height: 16),
                FilledButton.icon(
                  onPressed: _isLoading ? null : _recommend,
                  icon: _isLoading
                      ? const SizedBox(
                    width: 16, height: 16,
                    child: CircularProgressIndicator(strokeWidth: 2),
                  )
                      : const Icon(Icons.search),
                  label: const Text('Find Matching Jobs'),
                  style: FilledButton.styleFrom(
                    minimumSize: const Size(double.infinity, 50),
                    shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
                  ),
                ),
              ],
            ),
          ).animate().fadeIn(delay: 100.ms),

          // Error
          if (_error != null) ...[
            const SizedBox(height: 16),
            Container(
              padding: const EdgeInsets.all(14),
              decoration: BoxDecoration(
                color: cs.errorContainer,
                borderRadius: BorderRadius.circular(12),
              ),
              child: Text(_error!, style: TextStyle(color: cs.error)),
            ).animate().shake(),
          ],

          // Results
          if (_jobs.isNotEmpty) ...[
            const SizedBox(height: 28),
            Row(
              children: [
                Text(
                  'Top ${_jobs.length} Matches',
                  style: Theme.of(context).textTheme.titleMedium?.copyWith(
                    fontWeight: FontWeight.w700,
                  ),
                ),
                const Spacer(),
                Text(
                  'Based on $_querySkillCount skills',
                  style: TextStyle(color: cs.onSurfaceVariant, fontSize: 12),
                ),
              ],
            ).animate().fadeIn(delay: 150.ms),

            const SizedBox(height: 12),

            ..._jobs.asMap().entries.map((entry) {
              final job = entry.value;
              final rank = entry.key + 1;
              return _JobCard(job: job, rank: rank)
                  .animate()
                  .fadeIn(delay: Duration(milliseconds: 200 + rank * 80))
                  .slideX(begin: 0.05);
            }),
          ],
        ],
      ),
    );
  }
}

class _JobCard extends StatefulWidget {
  final JobMatch job;
  final int rank;

  const _JobCard({required this.job, required this.rank});

  @override
  State<_JobCard> createState() => _JobCardState();
}

class _JobCardState extends State<_JobCard> {
  bool _expanded = false;

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    final score = widget.job.matchScore;
    final skills = widget.job.skills;
    final certs = widget.job.certifications;
    final isTopMatch = widget.rank == 1;

    final scoreColor = switch (widget.job.scoreLevel) {
      ScoreLevel.high   => Colors.green,
      ScoreLevel.medium => Colors.orange,
      ScoreLevel.low    => cs.error,
    };

    return Container(
      margin: const EdgeInsets.only(bottom: 14),
      decoration: BoxDecoration(
        color: cs.surfaceContainerLow,
        borderRadius: BorderRadius.circular(16),
        border: isTopMatch ? Border.all(color: cs.primary, width: 2) : null,
        boxShadow: [
          BoxShadow(
            color: cs.shadow.withOpacity(0.06),
            blurRadius: 8,
            offset: const Offset(0, 2),
          ),
        ],
      ),
      child: Column(
        children: [
          // Header row
          InkWell(
            onTap: () => setState(() => _expanded = !_expanded),
            borderRadius: BorderRadius.circular(16),
            child: Padding(
              padding: const EdgeInsets.all(16),
              child: Row(
                children: [
                  // Rank badge
                  Container(
                    width: 40,
                    height: 40,
                    decoration: BoxDecoration(
                      color: isTopMatch ? cs.primary : cs.primaryContainer,
                      borderRadius: BorderRadius.circular(10),
                    ),
                    child: Center(
                      child: Text(
                        '#${widget.rank}',
                        style: TextStyle(
                          color: isTopMatch ? cs.onPrimary : cs.primary,
                          fontWeight: FontWeight.bold,
                          fontSize: 13,
                        ),
                      ),
                    ),
                  ),
                  const SizedBox(width: 14),

                  // Title + subtitle
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          widget.job['job_title'] ?? '',
                          style: const TextStyle(
                            fontWeight: FontWeight.w700,
                            fontSize: 15,
                          ),
                        ),
                        const SizedBox(height: 3),
                        Text(
                          '${skills.length} required skills',
                          style: TextStyle(
                            color: cs.onSurfaceVariant,
                            fontSize: 12,
                          ),
                        ),
                      ],
                    ),
                  ),

                  // Score badge
                  Container(
                    padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 7),
                    decoration: BoxDecoration(
                      color: scoreColor.withOpacity(0.12),
                      borderRadius: BorderRadius.circular(20),
                    ),
                    child: Text(
                      '${score.toStringAsFixed(0)}%',
                      style: TextStyle(
                        color: scoreColor,
                        fontWeight: FontWeight.bold,
                        fontSize: 14,
                      ),
                    ),
                  ),

                  const SizedBox(width: 8),
                  AnimatedRotation(
                    turns: _expanded ? 0.5 : 0,
                    duration: const Duration(milliseconds: 200),
                    child: Icon(Icons.keyboard_arrow_down, color: cs.onSurfaceVariant),
                  ),
                ],
              ),
            ),
          ),

          // Expanded details
          AnimatedCrossFade(
            firstChild: const SizedBox.shrink(),
            secondChild: Padding(
              padding: const EdgeInsets.fromLTRB(16, 0, 16, 16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Divider(height: 1),
                  const SizedBox(height: 14),

                  if (skills.isNotEmpty) ...[
                    Text(
                      'Required Skills',
                      style: TextStyle(
                        color: cs.onSurfaceVariant,
                        fontSize: 12,
                        fontWeight: FontWeight.w600,
                        letterSpacing: 0.5,
                      ),
                    ),
                    const SizedBox(height: 8),
                    Wrap(
                      spacing: 6,
                      runSpacing: 6,
                      children: skills
                          .map((s) => SkillChip(skill: s, type: SkillChipType.present))
                          .toList(),
                    ),
                    const SizedBox(height: 12),
                  ],

                  if (certs.isNotEmpty) ...[
                    Text(
                      'Certifications',
                      style: TextStyle(
                        color: cs.onSurfaceVariant,
                        fontSize: 12,
                        fontWeight: FontWeight.w600,
                        letterSpacing: 0.5,
                      ),
                    ),
                    const SizedBox(height: 8),
                    Wrap(
                      spacing: 6,
                      runSpacing: 6,
                      children: certs
                          .map((c) => SkillChip(skill: c, type: SkillChipType.cert))
                          .toList(),
                    ),
                  ],
                ],
              ),
            ),
            crossFadeState:
            _expanded ? CrossFadeState.showSecond : CrossFadeState.showFirst,
            duration: const Duration(milliseconds: 250),
          ),
        ],
      ),
    );
  }
}