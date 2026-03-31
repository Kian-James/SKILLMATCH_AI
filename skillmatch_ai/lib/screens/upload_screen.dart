// screens/upload_screen.dart
// Resume upload + full analysis. Shows extracted skills, certs, and top job matches.
// Saves a snapshot to Firestore via /analyze endpoint.

import 'dart:io';
import 'package:flutter/material.dart';
import 'package:file_picker/file_picker.dart';
import 'package:flutter_animate/flutter_animate.dart';
import 'package:provider/provider.dart';
import 'package:percent_indicator/circular_percent_indicator.dart';

import '../services/api_service.dart';
import '../services/auth_service.dart';
import '../widgets/skill_chip.dart';

class UploadScreen extends StatefulWidget {
  const UploadScreen({super.key});

  @override
  State<UploadScreen> createState() => _UploadScreenState();
}

class _UploadScreenState extends State<UploadScreen> {
  final _api = ApiService();
  bool _isAnalyzing = false;
  String? _error;
  Map<String, dynamic>? _result;
  String? _selectedFileName;

  Future<void> _pickAndAnalyze() async {
    final picked = await FilePicker.platform.pickFiles(
      type: FileType.custom,
      allowedExtensions: ['pdf', 'png', 'jpg', 'jpeg', 'bmp', 'tiff', 'webp'],
    );

    if (picked == null || picked.files.single.path == null) return;

    final file = File(picked.files.single.path!);
    setState(() {
      _isAnalyzing = true;
      _error = null;
      _result = null;
      _selectedFileName = picked.files.single.name;
    });

    try {
      final data = await _api.analyzeResume(file);
      setState(() => _result = data);
    } catch (e) {
      setState(() => _error = ApiService.extractError(e));
    } finally {
      setState(() => _isAnalyzing = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return SingleChildScrollView(
      padding: const EdgeInsets.all(20),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          _buildUploadCard(context),
          if (_isAnalyzing) ...[
            const SizedBox(height: 32),
            _buildLoadingState(),
          ],
          if (_error != null) ...[
            const SizedBox(height: 20),
            _buildErrorCard(context),
          ],
          if (_result != null) ...[
            const SizedBox(height: 24),
            _buildResultSection(context),
          ],
        ],
      ),
    );
  }

  Widget _buildUploadCard(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    return GestureDetector(
      onTap: _isAnalyzing ? null : _pickAndAnalyze,
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 200),
        padding: const EdgeInsets.symmetric(vertical: 40),
        decoration: BoxDecoration(
          color: cs.primaryContainer.withOpacity(0.4),
          borderRadius: BorderRadius.circular(20),
          border: Border.all(
            color: cs.primary.withOpacity(0.4),
            width: 2,
            style: BorderStyle.solid,
          ),
        ),
        child: Column(
          children: [
            Icon(Icons.cloud_upload_outlined, size: 56, color: cs.primary),
            const SizedBox(height: 16),
            Text(
              _selectedFileName != null
                  ? _selectedFileName!
                  : 'Tap to upload your resume',
              style: Theme.of(context).textTheme.titleMedium?.copyWith(
                    color: cs.primary,
                    fontWeight: FontWeight.w600,
                  ),
              textAlign: TextAlign.center,
            ),
            const SizedBox(height: 8),
            Text(
              'PDF, PNG, JPG, TIFF, BMP, WEBP — Max 10 MB',
              style: TextStyle(color: cs.onSurfaceVariant, fontSize: 12),
            ),
          ],
        ),
      ).animate().fadeIn().scale(begin: const Offset(0.97, 0.97)),
    );
  }

  Widget _buildLoadingState() {
    return Column(
      children: [
        const CircularProgressIndicator(),
        const SizedBox(height: 16),
        Text(
          'Analyzing your resume with AI...',
          style: TextStyle(color: Theme.of(context).colorScheme.onSurfaceVariant),
          textAlign: TextAlign.center,
        ),
        const SizedBox(height: 8),
        Text(
          'OCR → NLP → Embeddings → FAISS search',
          style: TextStyle(
            color: Theme.of(context).colorScheme.onSurfaceVariant,
            fontSize: 12,
          ),
          textAlign: TextAlign.center,
        ),
      ],
    ).animate().fadeIn();
  }

  Widget _buildErrorCard(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: cs.errorContainer,
        borderRadius: BorderRadius.circular(14),
      ),
      child: Row(
        children: [
          Icon(Icons.error_outline, color: cs.error),
          const SizedBox(width: 12),
          Expanded(
            child: Text(_error!, style: TextStyle(color: cs.error)),
          ),
        ],
      ),
    ).animate().shake();
  }

  Widget _buildResultSection(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    final skills = List<String>.from(_result!['skills'] ?? []);
    final certs = List<String>.from(_result!['certifications'] ?? []);
    final jobs = List<Map<String, dynamic>>.from(_result!['top_job_matches'] ?? []);
    final score = (_result!['best_match_score'] as num?)?.toDouble() ?? 0.0;

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        // Summary row
        Row(
          children: [
            _StatCard(label: 'Skills Found', value: '${skills.length}', icon: Icons.psychology_outlined),
            const SizedBox(width: 12),
            _StatCard(label: 'Certifications', value: '${certs.length}', icon: Icons.verified_outlined),
          ],
        ).animate().fadeIn(delay: 100.ms),

        const SizedBox(height: 20),

        // Best match score
        Center(
          child: CircularPercentIndicator(
            radius: 70,
            lineWidth: 10,
            percent: (score / 100).clamp(0.0, 1.0),
            center: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                Text(
                  '${score.toStringAsFixed(0)}%',
                  style: const TextStyle(fontSize: 24, fontWeight: FontWeight.bold),
                ),
                const Text('Best Match', style: TextStyle(fontSize: 11)),
              ],
            ),
            progressColor: cs.primary,
            backgroundColor: cs.primaryContainer,
            circularStrokeCap: CircularStrokeCap.round,
          ),
        ).animate().fadeIn(delay: 200.ms).scale(),

        const SizedBox(height: 24),

        // Skills
        if (skills.isNotEmpty) ...[
          _SectionTitle(title: 'Extracted Skills (${skills.length})'),
          const SizedBox(height: 10),
          Wrap(
            spacing: 8,
            runSpacing: 8,
            children: skills.map((s) => SkillChip(skill: s, type: SkillChipType.present)).toList(),
          ).animate().fadeIn(delay: 300.ms),
          const SizedBox(height: 20),
        ],

        // Certifications
        if (certs.isNotEmpty) ...[
          _SectionTitle(title: 'Certifications (${certs.length})'),
          const SizedBox(height: 10),
          Wrap(
            spacing: 8,
            runSpacing: 8,
            children: certs.map((c) => SkillChip(skill: c, type: SkillChipType.cert)).toList(),
          ).animate().fadeIn(delay: 350.ms),
          const SizedBox(height: 20),
        ],

        // Top job matches
        if (jobs.isNotEmpty) ...[
          _SectionTitle(title: 'Top Job Matches'),
          const SizedBox(height: 10),
          ...jobs.asMap().entries.map((entry) =>
            _JobMatchTile(job: entry.value, rank: entry.key + 1)
              .animate()
              .fadeIn(delay: Duration(milliseconds: 400 + entry.key * 80))
              .slideX(begin: 0.05),
          ),
        ],
      ],
    );
  }
}

// ── Supporting Widgets ─────────────────────────────────────────────────────────

class _StatCard extends StatelessWidget {
  final String label;
  final String value;
  final IconData icon;

  const _StatCard({required this.label, required this.value, required this.icon});

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    return Expanded(
      child: Container(
        padding: const EdgeInsets.all(16),
        decoration: BoxDecoration(
          color: cs.surfaceContainerHighest,
          borderRadius: BorderRadius.circular(14),
        ),
        child: Column(
          children: [
            Icon(icon, color: cs.primary, size: 28),
            const SizedBox(height: 8),
            Text(value, style: const TextStyle(fontSize: 24, fontWeight: FontWeight.bold)),
            Text(label, style: TextStyle(color: cs.onSurfaceVariant, fontSize: 12)),
          ],
        ),
      ),
    );
  }
}

class _SectionTitle extends StatelessWidget {
  final String title;
  const _SectionTitle({required this.title});

  @override
  Widget build(BuildContext context) {
    return Text(
      title,
      style: Theme.of(context).textTheme.titleMedium?.copyWith(
            fontWeight: FontWeight.w700,
          ),
    );
  }
}

class _JobMatchTile extends StatelessWidget {
  final Map<String, dynamic> job;
  final int rank;
  const _JobMatchTile({required this.job, required this.rank});

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    final score = (job['match_score'] as num?)?.toDouble() ?? 0;
    final skills = List<String>.from(job['skills'] ?? []);

    return Container(
      margin: const EdgeInsets.only(bottom: 12),
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: cs.surfaceContainerLow,
        borderRadius: BorderRadius.circular(14),
        border: rank == 1
            ? Border.all(color: cs.primary, width: 1.5)
            : null,
      ),
      child: Row(
        children: [
          CircleAvatar(
            radius: 22,
            backgroundColor: rank == 1 ? cs.primary : cs.primaryContainer,
            child: Text(
              '#$rank',
              style: TextStyle(
                color: rank == 1 ? cs.onPrimary : cs.primary,
                fontWeight: FontWeight.bold,
                fontSize: 13,
              ),
            ),
          ),
          const SizedBox(width: 14),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  job['job_title'] ?? '',
                  style: const TextStyle(fontWeight: FontWeight.w700, fontSize: 15),
                ),
                const SizedBox(height: 4),
                Text(
                  skills.take(3).join(' · '),
                  style: TextStyle(color: cs.onSurfaceVariant, fontSize: 12),
                  overflow: TextOverflow.ellipsis,
                ),
              ],
            ),
          ),
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
            decoration: BoxDecoration(
              color: _scoreColor(score).withOpacity(0.15),
              borderRadius: BorderRadius.circular(20),
            ),
            child: Text(
              '${score.toStringAsFixed(0)}%',
              style: TextStyle(
                color: _scoreColor(score),
                fontWeight: FontWeight.bold,
                fontSize: 13,
              ),
            ),
          ),
        ],
      ),
    );
  }

  Color _scoreColor(double score) {
    if (score >= 80) return Colors.green;
    if (score >= 60) return Colors.orange;
    return Colors.red;
  }
}
