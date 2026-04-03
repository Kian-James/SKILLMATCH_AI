// screens/upload_screen.dart
// Home page — 3 upload options (Scan · PDF · Image).
// After upload: Jobs and Skill Gap buttons appear.
// Jobs button → JobsScreen (top 5 matches).
// Skill Gap button → SkillGapScreen (target job + gap analysis).

import 'dart:io';
import 'package:flutter/material.dart';
import 'package:file_picker/file_picker.dart';
import 'package:image_picker/image_picker.dart';
import 'package:flutter_animate/flutter_animate.dart';

import '../models/resume_analysis.dart';
import '../services/api_service.dart';
import '../widgets/skill_chip.dart';
import 'jobs_screen.dart';
import 'skill_gap_screen.dart';

class UploadScreen extends StatefulWidget {
  const UploadScreen({super.key});

  @override
  State<UploadScreen> createState() => _UploadScreenState();
}

class _UploadScreenState extends State<UploadScreen> {
  final _api = ApiService();
  final _picker = ImagePicker();

  bool _isAnalyzing = false;
  String? _error;
  ResumeAnalysis? _result;
  String? _selectedFileName;

  // ── Upload options ──────────────────────────────────────────────────────────

  Future<void> _scanDocument() async {
    final XFile? photo = await _picker.pickImage(
      source: ImageSource.camera,
      imageQuality: 90,
      preferredCameraDevice: CameraDevice.rear,
    );
    if (photo == null) return;
    await _analyze(File(photo.path), photo.name);
  }

  Future<void> _uploadPdf() async {
    final picked = await FilePicker.platform.pickFiles(
      type: FileType.custom,
      allowedExtensions: ['pdf'],
    );
    if (picked == null || picked.files.single.path == null) return;
    await _analyze(
        File(picked.files.single.path!), picked.files.single.name);
  }

  Future<void> _uploadImage() async {
    final picked = await FilePicker.platform.pickFiles(
      type: FileType.custom,
      allowedExtensions: ['png', 'jpg', 'jpeg', 'bmp', 'tiff', 'webp'],
    );
    if (picked == null || picked.files.single.path == null) return;
    await _analyze(
        File(picked.files.single.path!), picked.files.single.name);
  }

  Future<void> _analyze(File file, String fileName) async {
    setState(() {
      _isAnalyzing = true;
      _error = null;
      _result = null;
      _selectedFileName = fileName;
    });
    try {
      final result = await _api.analyzeResume(file);
      setState(() => _result = result);
    } catch (e) {
      setState(() => _error = ApiService.extractError(e));
    } finally {
      setState(() => _isAnalyzing = false);
    }
  }

  // ── Navigation ──────────────────────────────────────────────────────────────

  void _goToJobs() {
    if (_result == null) return;
    Navigator.push(context,
        MaterialPageRoute(builder: (_) => JobsScreen(analysis: _result!)));
  }

  void _goToSkillGap() {
    if (_result == null) return;
    Navigator.push(context,
        MaterialPageRoute(builder: (_) => SkillGapScreen(analysis: _result!)));
  }

  // ── UI ──────────────────────────────────────────────────────────────────────

  @override
  Widget build(BuildContext context) {
    return SingleChildScrollView(
      padding: const EdgeInsets.all(20),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          _buildHeader(context),
          const SizedBox(height: 24),
          _buildUploadOptions(context),
          if (_isAnalyzing) ...[
            const SizedBox(height: 28),
            _buildLoading(context),
          ],
          if (_error != null) ...[
            const SizedBox(height: 20),
            _buildError(context),
          ],
          if (_result != null) ...[
            const SizedBox(height: 28),
            _buildActionButtons(context),
            const SizedBox(height: 24),
            _buildSummary(context),
          ],
        ],
      ),
    );
  }

  Widget _buildHeader(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          _result == null ? 'Upload Your Resume' : 'Resume Analyzed ✓',
          style: Theme.of(context).textTheme.headlineSmall?.copyWith(
            fontWeight: FontWeight.w800,
            letterSpacing: -0.5,
          ),
        ),
        const SizedBox(height: 6),
        Text(
          _result == null
              ? 'Scan a paper, upload a PDF, or pick an image.'
              : 'Tap Jobs or Skill Gap below to explore your results.',
          style: TextStyle(color: cs.onSurfaceVariant, fontSize: 14),
        ),
      ],
    ).animate().fadeIn();
  }

  Widget _buildUploadOptions(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    return Row(
      children: [
        _UploadCard(
          icon: Icons.document_scanner_outlined,
          label: 'Scan',
          color: cs.primaryContainer,
          iconColor: cs.primary,
          onTap: _isAnalyzing ? null : _scanDocument,
        ),
        const SizedBox(width: 12),
        _UploadCard(
          icon: Icons.picture_as_pdf_outlined,
          label: 'PDF',
          color: cs.secondaryContainer,
          iconColor: cs.secondary,
          onTap: _isAnalyzing ? null : _uploadPdf,
        ),
        const SizedBox(width: 12),
        _UploadCard(
          icon: Icons.image_outlined,
          label: 'Image',
          color: cs.tertiaryContainer,
          iconColor: cs.tertiary,
          onTap: _isAnalyzing ? null : _uploadImage,
        ),
      ],
    ).animate().fadeIn(delay: 100.ms);
  }

  Widget _buildLoading(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    return Container(
      padding: const EdgeInsets.all(24),
      decoration: BoxDecoration(
        color: cs.surfaceContainerLow,
        borderRadius: BorderRadius.circular(16),
      ),
      child: Column(
        children: [
          const CircularProgressIndicator(),
          const SizedBox(height: 16),
          Text(
            'Analyzing $_selectedFileName...',
            style: const TextStyle(fontWeight: FontWeight.w600),
            textAlign: TextAlign.center,
          ),
          const SizedBox(height: 8),
          Text(
            'OCR → NLP → Embeddings → FAISS',
            style: TextStyle(color: cs.onSurfaceVariant, fontSize: 12),
          ),
        ],
      ),
    ).animate().fadeIn();
  }

  Widget _buildError(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
          color: cs.errorContainer, borderRadius: BorderRadius.circular(14)),
      child: Row(
        children: [
          Icon(Icons.error_outline, color: cs.error),
          const SizedBox(width: 12),
          Expanded(child: Text(_error!, style: TextStyle(color: cs.error))),
        ],
      ),
    ).animate().shake();
  }

  // Jobs + Skill Gap buttons — only visible after upload
  Widget _buildActionButtons(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    return Row(
      children: [
        Expanded(
          child: FilledButton.icon(
            onPressed: _goToJobs,
            icon: const Icon(Icons.work_outline),
            label: const Text('Jobs'),
            style: FilledButton.styleFrom(
              padding: const EdgeInsets.symmetric(vertical: 16),
              shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(14)),
            ),
          ),
        ),
        const SizedBox(width: 12),
        Expanded(
          child: FilledButton.icon(
            onPressed: _goToSkillGap,
            icon: const Icon(Icons.track_changes_outlined),
            label: const Text('Skill Gap'),
            style: FilledButton.styleFrom(
              padding: const EdgeInsets.symmetric(vertical: 16),
              backgroundColor: cs.secondary,
              shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(14)),
            ),
          ),
        ),
      ],
    ).animate().fadeIn(delay: 100.ms).slideY(begin: 0.1);
  }

  Widget _buildSummary(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    final r = _result!;

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        // Quick stats
        Row(
          children: [
            _StatCard(
                label: 'Skills',
                value: '${r.skills.length}',
                icon: Icons.psychology_outlined),
            const SizedBox(width: 12),
            _StatCard(
                label: 'Certs',
                value: '${r.certifications.length}',
                icon: Icons.verified_outlined),
            const SizedBox(width: 12),
            Expanded(
              child: Container(
                padding: const EdgeInsets.all(16),
                decoration: BoxDecoration(
                  color: cs.primaryContainer.withOpacity(0.5),
                  borderRadius: BorderRadius.circular(14),
                ),
                child: Column(
                  children: [
                    Text(
                      '${r.bestMatchScore.toStringAsFixed(0)}%',
                      style: TextStyle(
                        fontSize: 20,
                        fontWeight: FontWeight.bold,
                        color: cs.primary,
                      ),
                    ),
                    Text('Best Match',
                        style: TextStyle(
                            color: cs.onSurfaceVariant, fontSize: 11)),
                  ],
                ),
              ),
            ),
          ],
        ).animate().fadeIn(delay: 150.ms),

        // Skills
        if (r.skills.isNotEmpty) ...[
          const SizedBox(height: 20),
          Text('Extracted Skills',
              style: Theme.of(context)
                  .textTheme
                  .titleSmall
                  ?.copyWith(fontWeight: FontWeight.w700)),
          const SizedBox(height: 10),
          Wrap(
            spacing: 8,
            runSpacing: 8,
            children: r.skills
                .map((s) => SkillChip(skill: s, type: SkillChipType.present))
                .toList(),
          ).animate().fadeIn(delay: 200.ms),
        ],

        // Certs
        if (r.certifications.isNotEmpty) ...[
          const SizedBox(height: 16),
          Text('Certifications',
              style: Theme.of(context)
                  .textTheme
                  .titleSmall
                  ?.copyWith(fontWeight: FontWeight.w700)),
          const SizedBox(height: 10),
          Wrap(
            spacing: 8,
            runSpacing: 8,
            children: r.certifications
                .map((c) => SkillChip(skill: c, type: SkillChipType.cert))
                .toList(),
          ).animate().fadeIn(delay: 250.ms),
        ],

        const SizedBox(height: 16),
        Container(
          padding: const EdgeInsets.all(14),
          decoration: BoxDecoration(
            color: cs.surfaceContainerLow,
            borderRadius: BorderRadius.circular(12),
          ),
          child: Row(
            children: [
              Icon(Icons.lightbulb_outline, color: cs.primary, size: 18),
              const SizedBox(width: 10),
              Expanded(
                child: Text(
                  'Tap Jobs to see your top 5 matches, or Skill Gap to find what you\'re missing for a specific role.',
                  style:
                  TextStyle(color: cs.onSurfaceVariant, fontSize: 13),
                ),
              ),
            ],
          ),
        ).animate().fadeIn(delay: 300.ms),
      ],
    );
  }
}

// ── Widgets ───────────────────────────────────────────────────────────────────

class _UploadCard extends StatelessWidget {
  final IconData icon;
  final String label;
  final Color color;
  final Color iconColor;
  final VoidCallback? onTap;

  const _UploadCard({
    required this.icon,
    required this.label,
    required this.color,
    required this.iconColor,
    this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return Expanded(
      child: GestureDetector(
        onTap: onTap,
        child: AnimatedOpacity(
          opacity: onTap == null ? 0.5 : 1.0,
          duration: const Duration(milliseconds: 200),
          child: Container(
            padding: const EdgeInsets.symmetric(vertical: 22),
            decoration: BoxDecoration(
              color: color,
              borderRadius: BorderRadius.circular(16),
            ),
            child: Column(
              children: [
                Icon(icon, size: 32, color: iconColor),
                const SizedBox(height: 8),
                Text(
                  label,
                  style: TextStyle(
                    color: iconColor,
                    fontWeight: FontWeight.w600,
                    fontSize: 13,
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}

class _StatCard extends StatelessWidget {
  final String label;
  final String value;
  final IconData icon;
  const _StatCard(
      {required this.label, required this.value, required this.icon});

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
            Icon(icon, color: cs.primary, size: 22),
            const SizedBox(height: 6),
            Text(value,
                style: const TextStyle(
                    fontSize: 20, fontWeight: FontWeight.bold)),
            Text(label,
                style: TextStyle(
                    color: cs.onSurfaceVariant, fontSize: 11)),
          ],
        ),
      ),
    );
  }
}