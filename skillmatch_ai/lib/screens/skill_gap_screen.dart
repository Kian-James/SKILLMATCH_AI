// screens/skill_gap_screen.dart
import 'package:flutter/material.dart';
import 'package:flutter_animate/flutter_animate.dart';
import '../models/skill_gap.dart';
import '../services/api_service.dart';
import '../widgets/skill_chip.dart';

class SkillGapScreen extends StatefulWidget {
  const SkillGapScreen({super.key});
  @override
  State<SkillGapScreen> createState() => _SkillGapScreenState();
}

class _SkillGapScreenState extends State<SkillGapScreen> {
  final _api = ApiService();
  final _jobCtrl = TextEditingController();
  final _skillsCtrl = TextEditingController();
  bool _isLoading = false;
  String? _error;
  SkillGapResult? _result;

  Future<void> _analyze() async {
    final job = _jobCtrl.text.trim();
    final rawSkills = _skillsCtrl.text.trim();
    if (job.isEmpty || rawSkills.isEmpty) return;

    final skills = rawSkills.split(',').map((s) => s.trim()).where((s) => s.isNotEmpty).toList();

    setState(() { _isLoading = true; _error = null; _result = null; });
    try {
      final data = await _api.skillGap(targetJob: job, userSkills: skills);
      setState(() => _result = data);
    } catch (e) {
      setState(() => _error = ApiService.extractError(e));
    } finally {
      setState(() => _isLoading = false);
    }
  }

  void _showResources(String skill) {
    showModalBottomSheet(  // ignore: unawaited_futures
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
    return SingleChildScrollView(
      padding: const EdgeInsets.all(20),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          // Input form
          Container(
            padding: const EdgeInsets.all(20),
            decoration: BoxDecoration(
              color: cs.surfaceContainerLow,
              borderRadius: BorderRadius.circular(16),
            ),
            child: Column(
              children: [
                TextField(
                  controller: _jobCtrl,
                  decoration: const InputDecoration(
                    labelText: 'Target Job Title',
                    hintText: 'e.g., Software Engineer',
                    prefixIcon: Icon(Icons.work_outline),
                  ),
                ),
                const SizedBox(height: 16),
                TextField(
                  controller: _skillsCtrl,
                  maxLines: 3,
                  decoration: const InputDecoration(
                    labelText: 'Your Skills (comma-separated)',
                    hintText: 'Python, Docker, React...',
                    prefixIcon: Icon(Icons.psychology_outlined),
                    alignLabelWithHint: true,
                  ),
                ),
                const SizedBox(height: 16),
                FilledButton.icon(
                  onPressed: _isLoading ? null : _analyze,
                  icon: _isLoading
                      ? const SizedBox(width: 16, height: 16, child: CircularProgressIndicator(strokeWidth: 2))
                      : const Icon(Icons.search),
                  label: const Text('Analyze Gap'),
                  style: FilledButton.styleFrom(minimumSize: const Size(double.infinity, 48)),
                ),
              ],
            ),
          ).animate().fadeIn(),

          if (_error != null) ...[
            const SizedBox(height: 16),
            Container(
              padding: const EdgeInsets.all(14),
              decoration: BoxDecoration(color: cs.errorContainer, borderRadius: BorderRadius.circular(12)),
              child: Text(_error!, style: TextStyle(color: cs.error)),
            ),
          ],

          if (_result != null) ...[
            const SizedBox(height: 24),
            _buildResults(context),
          ],
        ],
      ),
    );
  }

  Widget _buildResults(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    final r = _result!;
    final score = r.completionScore;
    final matched = r.matchedSkills;
    final missing = r.missingSkills;
    final msg = r.message;

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        // Score card
        Container(
          padding: const EdgeInsets.all(20),
          decoration: BoxDecoration(
            gradient: LinearGradient(
              colors: [cs.primaryContainer, cs.secondaryContainer],
              begin: Alignment.topLeft, end: Alignment.bottomRight,
            ),
            borderRadius: BorderRadius.circular(16),
          ),
          child: Column(
            children: [
              Text(r.targetJob, style: const TextStyle(fontSize: 18, fontWeight: FontWeight.bold)),
              const SizedBox(height: 16),
              Stack(
                alignment: Alignment.center,
                children: [
                  SizedBox(
                    width: 100, height: 100,
                    child: CircularProgressIndicator(
                      value: score / 100,
                      strokeWidth: 10,
                      color: cs.primary,
                      backgroundColor: cs.surface.withOpacity(0.4),
                      strokeCap: StrokeCap.round,
                    ),
                  ),
                  Text('${score.toStringAsFixed(0)}%', style: const TextStyle(fontSize: 26, fontWeight: FontWeight.bold)),
                ],
              ),
              const SizedBox(height: 12),
              Text(msg, textAlign: TextAlign.center, style: const TextStyle(fontSize: 13)),
              const SizedBox(height: 8),
              Text('${matched.length} of ${r.totalRequired} required skills matched',
                  style: TextStyle(color: cs.onSurfaceVariant, fontSize: 12)),
            ],
          ),
        ).animate().fadeIn().scale(begin: const Offset(0.95, 0.95)),

        const SizedBox(height: 20),

        // Matched skills
        if (matched.isNotEmpty) ...[
          Text('✅ Skills You Have (${matched.length})',
              style: const TextStyle(fontWeight: FontWeight.w700, fontSize: 15)),
          const SizedBox(height: 10),
          Wrap(spacing: 8, runSpacing: 8,
            children: matched.map((s) => SkillChip(skill: s, type: SkillChipType.present)).toList(),
          ).animate().fadeIn(delay: 200.ms),
          const SizedBox(height: 20),
        ],

        // Missing skills — tappable to show resources
        if (missing.isNotEmpty) ...[
          Text('❌ Missing Skills (${missing.length}) — Tap for resources',
              style: const TextStyle(fontWeight: FontWeight.w700, fontSize: 15)),
          const SizedBox(height: 10),
          Wrap(spacing: 8, runSpacing: 8,
            children: missing.map((s) => SkillChip(
              skill: s,
              type: SkillChipType.missing,
              onTap: () => _showResources(s),
            )).toList(),
          ).animate().fadeIn(delay: 300.ms),
        ],
      ],
    );
  }
}

// ─── Resources Bottom Sheet ───────────────────────────────────────────────────

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
  void initState() {
    super.initState();
    _load();
  }

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
      maxChildSize: 0.9,
      minChildSize: 0.4,
      expand: false,
      builder: (_, scrollCtrl) => Container(
        padding: const EdgeInsets.all(20),
        child: Column(
          children: [
            Container(width: 40, height: 4, margin: const EdgeInsets.only(bottom: 16),
                decoration: BoxDecoration(color: Colors.grey[300], borderRadius: BorderRadius.circular(2))),
            Text('Learn: ${widget.skill}',
                style: const TextStyle(fontSize: 18, fontWeight: FontWeight.bold)),
            const SizedBox(height: 4),
            Text('AI-generated learning resources', style: TextStyle(color: cs.onSurfaceVariant, fontSize: 13)),
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
                      title: Text(r.title, style: const TextStyle(fontWeight: FontWeight.w600)),
                      subtitle: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text('${r.platform} • ${r.duration}',
                              style: TextStyle(color: cs.primary, fontSize: 12, fontWeight: FontWeight.w500)),
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