// screens/skill_gap_screen.dart
// Skill gap flow: user already uploaded resume on home page.
// They type a target job → app compares their skills vs job requirements.
// Shows missing skills (tappable for learning resources) + certs needed.

import 'package:flutter/material.dart';
import 'package:flutter_animate/flutter_animate.dart';

import '../models/resume_analysis.dart';
import '../models/skill_gap.dart';
import '../services/api_service.dart';
import '../widgets/skill_chip.dart';

class SkillGapScreen extends StatefulWidget {
  final ResumeAnalysis analysis;

  const SkillGapScreen({super.key, required this.analysis});

  @override
  State<SkillGapScreen> createState() => _SkillGapScreenState();
}

class _SkillGapScreenState extends State<SkillGapScreen> {
  final _api = ApiService();
  final _jobCtrl = TextEditingController();

  bool _isLoading = false;
  String? _error;
  SkillGapResult? _result;

  @override
  void dispose() {
    _jobCtrl.dispose();
    super.dispose();
  }

  Future<void> _analyze() async {
    final job = _jobCtrl.text.trim();
    if (job.isEmpty) {
      setState(() => _error = 'Please enter a target job title.');
      return;
    }
    setState(() { _isLoading = true; _error = null; _result = null; });

    try {
      final result = await _api.skillGap(
        targetJob: job,
        userSkills: widget.analysis.skills,
        userCertifications: widget.analysis.certifications,
      );
      setState(() => _result = result);
    } catch (e) {
      setState(() => _error = ApiService.extractError(e));
    } finally {
      setState(() => _isLoading = false);
    }
  }

  void _showResources(String skill) {
    showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(20)),
      ),
      builder: (_) => _ResourcesSheet(skill: skill, api: _api),
    );
  }

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;

    return Scaffold(
      appBar: AppBar(
        title: const Text('Skill Gap Analysis'),
        backgroundColor: Colors.transparent,
        elevation: 0,
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(20),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            // Resume context banner
            _buildResumeContext(context),
            const SizedBox(height: 24),

            // Target job input
            Text(
              'What job do you want?',
              style: Theme.of(context)
                  .textTheme
                  .titleMedium
                  ?.copyWith(fontWeight: FontWeight.w700),
            ),
            const SizedBox(height: 10),
            TextField(
              controller: _jobCtrl,
              textCapitalization: TextCapitalization.words,
              decoration: const InputDecoration(
                hintText: 'e.g., Software Engineer, Data Scientist...',
                prefixIcon: Icon(Icons.work_outline),
              ),
              onSubmitted: (_) => _analyze(),
            ).animate().fadeIn(delay: 100.ms),

            const SizedBox(height: 20),

            FilledButton.icon(
              onPressed: _isLoading ? null : _analyze,
              icon: _isLoading
                  ? const SizedBox(
                  width: 18, height: 18,
                  child: CircularProgressIndicator(strokeWidth: 2))
                  : const Icon(Icons.search),
              label: Text(_isLoading ? 'Analyzing...' : 'Detect Skill Gap'),
              style: FilledButton.styleFrom(
                padding: const EdgeInsets.symmetric(vertical: 16),
                shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(14)),
              ),
            ).animate().fadeIn(delay: 150.ms),

            if (_error != null) ...[
              const SizedBox(height: 16),
              Container(
                padding: const EdgeInsets.all(14),
                decoration: BoxDecoration(
                    color: cs.errorContainer,
                    borderRadius: BorderRadius.circular(12)),
                child: Text(_error!, style: TextStyle(color: cs.error)),
              ),
            ],

            if (_result != null) ...[
              const SizedBox(height: 28),
              _buildResults(context),
            ],
          ],
        ),
      ),
    );
  }

  Widget _buildResumeContext(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    final skills = widget.analysis.skills;

    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: cs.secondaryContainer.withOpacity(0.5),
        borderRadius: BorderRadius.circular(14),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(children: [
            Icon(Icons.check_circle_outline, color: cs.secondary, size: 18),
            const SizedBox(width: 8),
            Text(
              'Resume loaded — ${skills.length} skills detected',
              style: TextStyle(
                  color: cs.onSecondaryContainer,
                  fontWeight: FontWeight.w600,
                  fontSize: 13),
            ),
          ]),
          if (skills.isNotEmpty) ...[
            const SizedBox(height: 10),
            Wrap(
              spacing: 6, runSpacing: 6,
              children: skills.take(8)
                  .map((s) => SkillChip(skill: s, type: SkillChipType.present))
                  .toList(),
            ),
            if (skills.length > 8)
              Padding(
                padding: const EdgeInsets.only(top: 6),
                child: Text('+${skills.length - 8} more skills',
                    style: TextStyle(color: cs.onSecondaryContainer, fontSize: 12)),
              ),
          ],
        ],
      ),
    ).animate().fadeIn();
  }

  Widget _buildResults(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    final r = _result!;

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        // Score card
        Container(
          width: double.infinity,
          padding: const EdgeInsets.all(24),
          decoration: BoxDecoration(
            gradient: LinearGradient(
              colors: [cs.primaryContainer, cs.secondaryContainer],
              begin: Alignment.topLeft, end: Alignment.bottomRight,
            ),
            borderRadius: BorderRadius.circular(18),
          ),
          child: Column(
            children: [
              Text(r.targetJob,
                  style: const TextStyle(fontSize: 20, fontWeight: FontWeight.w800),
                  textAlign: TextAlign.center),
              const SizedBox(height: 20),
              Stack(
                alignment: Alignment.center,
                children: [
                  SizedBox(
                    width: 110, height: 110,
                    child: CircularProgressIndicator(
                      value: r.completionFraction,
                      strokeWidth: 10,
                      color: cs.primary,
                      backgroundColor: cs.surface.withOpacity(0.4),
                      strokeCap: StrokeCap.round,
                    ),
                  ),
                  Column(children: [
                    Text('${r.completionScore.toStringAsFixed(0)}%',
                        style: const TextStyle(fontSize: 28, fontWeight: FontWeight.bold)),
                    const Text('ready', style: TextStyle(fontSize: 12)),
                  ]),
                ],
              ),
              const SizedBox(height: 16),
              Text(r.message, textAlign: TextAlign.center, style: const TextStyle(fontSize: 14)),
              const SizedBox(height: 8),
              Text('${r.matchCount} of ${r.totalRequired} required skills matched',
                  style: TextStyle(color: cs.onSurfaceVariant, fontSize: 13)),
            ],
          ),
        ).animate().fadeIn().scale(begin: const Offset(0.95, 0.95)),

        // Matched skills
        if (r.matchedSkills.isNotEmpty) ...[
          const SizedBox(height: 24),
          Text('✅ Skills You Have (${r.matchedSkills.length})',
              style: const TextStyle(fontWeight: FontWeight.w700, fontSize: 16)),
          const SizedBox(height: 10),
          Wrap(spacing: 8, runSpacing: 8,
            children: r.matchedSkills
                .map((s) => SkillChip(skill: s, type: SkillChipType.present))
                .toList(),
          ).animate().fadeIn(delay: 150.ms),
        ],

        // Missing skills — tappable
        if (r.missingSkills.isNotEmpty) ...[
          const SizedBox(height: 24),
          Text('❌ Missing Skills (${r.missingSkills.length})',
              style: const TextStyle(fontWeight: FontWeight.w700, fontSize: 16)),
          const SizedBox(height: 6),
          Text('Tap any skill to see learning resources.',
              style: TextStyle(color: cs.onSurfaceVariant, fontSize: 13)),
          const SizedBox(height: 10),
          Wrap(spacing: 8, runSpacing: 8,
            children: r.missingSkills
                .map((s) => SkillChip(
              skill: s,
              type: SkillChipType.missing,
              onTap: () => _showResources(s),
            ))
                .toList(),
          ).animate().fadeIn(delay: 200.ms),
        ],

        // Required certs
        if (r.requiredCertifications.isNotEmpty) ...[
          const SizedBox(height: 24),
          Text('🏅 Recommended Certifications',
              style: const TextStyle(fontWeight: FontWeight.w700, fontSize: 16)),
          const SizedBox(height: 10),
          Wrap(spacing: 8, runSpacing: 8,
            children: r.requiredCertifications
                .map((c) => SkillChip(skill: c, type: SkillChipType.cert))
                .toList(),
          ).animate().fadeIn(delay: 250.ms),
        ],
      ],
    );
  }
}

// ── Resources Bottom Sheet ────────────────────────────────────────────────────

class _ResourcesSheet extends StatefulWidget {
  final String skill;
  final ApiService api;
  const _ResourcesSheet({required this.skill, required this.api});

  @override
  State<_ResourcesSheet> createState() => _ResourcesSheetState();
}

class _ResourcesSheetState extends State<_ResourcesSheet> {
  bool _loading = true;
  List<LearningResource> _resources = [];
  String? _error;

  @override
  void initState() { super.initState(); _load(); }

  Future<void> _load() async {
    try {
      final resources = await widget.api.skillResources(widget.skill);
      setState(() { _resources = resources; _loading = false; });
    } catch (e) {
      setState(() { _error = ApiService.extractError(e); _loading = false; });
    }
  }

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    return DraggableScrollableSheet(
      initialChildSize: 0.6,
      maxChildSize: 0.92,
      minChildSize: 0.4,
      expand: false,
      builder: (_, scrollCtrl) => Container(
        padding: const EdgeInsets.all(20),
        child: Column(
          children: [
            Container(
              width: 40, height: 4,
              margin: const EdgeInsets.only(bottom: 16),
              decoration: BoxDecoration(
                  color: Colors.grey[300], borderRadius: BorderRadius.circular(2)),
            ),
            Text('Learn: ${widget.skill}',
                style: const TextStyle(fontSize: 18, fontWeight: FontWeight.bold)),
            const SizedBox(height: 4),
            Text('AI-generated learning resources',
                style: TextStyle(color: cs.onSurfaceVariant, fontSize: 13)),
            const SizedBox(height: 16),
            if (_loading) const Expanded(child: Center(child: CircularProgressIndicator())),
            if (_error != null) Text(_error!, style: TextStyle(color: cs.error)),
            if (!_loading && _error == null)
              Expanded(
                child: ListView.separated(
                  controller: scrollCtrl,
                  itemCount: _resources.length,
                  separatorBuilder: (_, __) => const Divider(height: 1),
                  itemBuilder: (_, i) {
                    final r = _resources[i];
                    return ListTile(
                      leading: CircleAvatar(
                        backgroundColor: cs.primaryContainer,
                        child: Text(r.emoji, style: const TextStyle(fontSize: 18)),
                      ),
                      title: Text(r.title,
                          style: const TextStyle(fontWeight: FontWeight.w600)),
                      subtitle: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text('${r.platform} · ${r.duration}',
                              style: TextStyle(
                                  color: cs.primary, fontSize: 12,
                                  fontWeight: FontWeight.w500)),
                          Text(r.description, style: const TextStyle(fontSize: 12)),
                        ],
                      ),
                      isThreeLine: true,
                    );
                  },
                ),
              ),
          ],
        ),
      ),
    );
  }
}