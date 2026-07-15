// lib/screens/results_screen.dart

import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:intl/intl.dart';

import '../models/models.dart';
import '../services/firestore_service.dart';
import '../theme.dart';
import '../widgets/common_widgets.dart';

class ResultsScreen extends StatefulWidget {
  final ResumeScanRecord record;
  final ResumeUser user;
  final bool fromCache;

  const ResultsScreen({
    super.key,
    required this.record,
    required this.user,
    required this.fromCache,
  });

  @override
  State<ResultsScreen> createState() => _ResultsScreenState();
}

class _ResultsScreenState extends State<ResultsScreen> {
  late List<String> _missingSkills;
  final _addSkillCtrl = TextEditingController();
  final _firestore = FirestoreService();
  bool _savingSkills = false;

  double get _accuracyValue {
    final clean = widget.record.nlpAccuracy.replaceAll('%', '').trim();
    return double.tryParse(clean) ?? 0;
  }

  @override
  void initState() {
    super.initState();
    _missingSkills = List<String>.from(widget.record.missingSkills);
  }

  @override
  void dispose() {
    _addSkillCtrl.dispose();
    super.dispose();
  }

  Future<void> _persistMissingSkills() async {
    setState(() => _savingSkills = true);
    try {
      await _firestore.updateMissingSkills(widget.record.id, _missingSkills);
    } catch (_) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(
              content: Text('Could not save that change — try again.')),
        );
      }
    } finally {
      if (mounted) setState(() => _savingSkills = false);
    }
  }

  void _addSkill() {
    final skill = _addSkillCtrl.text.trim();
    if (skill.isEmpty) return;
    if (_missingSkills
        .map((s) => s.toLowerCase())
        .contains(skill.toLowerCase())) {
      _addSkillCtrl.clear();
      return;
    }
    setState(() {
      _missingSkills.add(skill);
      _addSkillCtrl.clear();
    });
    _persistMissingSkills();
  }

  void _removeSkill(String skill) {
    setState(() => _missingSkills.remove(skill));
    _persistMissingSkills();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text(
          widget.record.targetJobTitle,
          overflow: TextOverflow.ellipsis,
        ),
        actions: [
          if (widget.fromCache)
            Container(
              margin: const EdgeInsets.only(right: 16),
              child: const TealBadge(text: 'Cached'),
            ),
        ],
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            _buildJobHeader(context),
            const SizedBox(height: 20),
            _buildStatRow(context),
            const SizedBox(height: 24),
            _buildAccuracyCard(context),
            const SizedBox(height: 24),
            _buildSkillsSection(context, 'Matched Skills',
                widget.record.matchedSkills, SkillChipType.matched,
                icon: Icons.check_circle_outline_rounded,
                iconColor: AppTheme.green500),
            const SizedBox(height: 20),
            _buildMissingSkillsSection(context),
            const SizedBox(height: 20),
            _buildAllSkillsSection(context),
            const SizedBox(height: 20),
            if (widget.record.certCount > 0) _buildCertSection(context),
            const SizedBox(height: 20),
            _buildMetaCard(context),
            const SizedBox(height: 32),
          ],
        ),
      ),
    );
  }

  Widget _buildJobHeader(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(18),
      decoration: BoxDecoration(
        color: AppColors.of(context).navy800,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: AppTheme.teal500.withOpacity(0.25)),
      ),
      child: Row(
        children: [
          Container(
            width: 48,
            height: 48,
            decoration: BoxDecoration(
              color: AppTheme.teal500.withOpacity(0.1),
              borderRadius: BorderRadius.circular(12),
            ),
            child: const Icon(Icons.work_outline_rounded,
                color: AppTheme.teal500, size: 22),
          ),
          const SizedBox(width: 14),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(widget.record.targetJobTitle,
                    style: Theme.of(context).textTheme.titleLarge),
                const SizedBox(height: 3),
                Text(
                    widget.record.industry.isNotEmpty
                        ? widget.record.industry
                        : 'Industry',
                    style: Theme.of(context).textTheme.bodyMedium),
                const SizedBox(height: 6),
                Text(widget.record.fileName,
                    style: GoogleFonts.workSans(
                        fontSize: 11,
                        color: AppColors.of(context).textTertiary,
                        fontStyle: FontStyle.italic)),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildStatRow(BuildContext context) {
    return GridView.count(
      crossAxisCount: 2,
      crossAxisSpacing: 10,
      mainAxisSpacing: 10,
      shrinkWrap: true,
      physics: const NeverScrollableScrollPhysics(),
      childAspectRatio: 1.5,
      children: [
        StatCard(
          label: 'Total Skills',
          value: '${widget.record.skillCount}',
          valueColor: AppColors.of(context).textPrimary,
          icon: Icons.psychology_outlined,
        ),
        StatCard(
          label: 'Matched',
          value: '${widget.record.matchedCount}',
          valueColor: AppTheme.green500,
          icon: Icons.check_circle_outline_rounded,
        ),
        StatCard(
          label: 'Missing',
          value: '${_missingSkills.length}',
          valueColor: AppTheme.red500,
          icon: Icons.remove_circle_outline_rounded,
        ),
        StatCard(
          label: 'Certifications',
          value: '${widget.record.certCount}',
          valueColor: AppTheme.amber500,
          icon: Icons.verified_outlined,
        ),
      ],
    );
  }

  Widget _buildAccuracyCard(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: AppColors.of(context).navy800,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: AppColors.of(context).cardBorder),
      ),
      child: Row(
        children: [
          AccuracyRing(accuracy: widget.record.nlpAccuracy, size: 80),
          const SizedBox(width: 20),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text('NLP Match Accuracy',
                    style: Theme.of(context).textTheme.titleMedium),
                const SizedBox(height: 6),
                Text(
                  _accuracyDescription,
                  style: Theme.of(context).textTheme.bodyMedium,
                ),
                const SizedBox(height: 10),
                ClipRRect(
                  borderRadius: BorderRadius.circular(4),
                  child: LinearProgressIndicator(
                    value: _accuracyValue / 100,
                    minHeight: 6,
                    backgroundColor: AppColors.of(context).navy600,
                    valueColor: AlwaysStoppedAnimation<Color>(_accuracyColor),
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  String get _accuracyDescription {
    if (_accuracyValue >= 75) return 'Strong match for this role';
    if (_accuracyValue >= 50) return 'Partial match — a few gaps to fill';
    return 'Needs work — focus on missing skills';
  }

  Color get _accuracyColor {
    if (_accuracyValue >= 75) return AppTheme.green500;
    if (_accuracyValue >= 50) return AppTheme.amber500;
    return AppTheme.red500;
  }

  Widget _buildSkillsSection(
    BuildContext context,
    String title,
    List<String> skills,
    SkillChipType type, {
    required IconData icon,
    required Color iconColor,
    String? subtitle,
  }) {
    if (skills.isEmpty) return const SizedBox.shrink();

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          children: [
            Icon(icon, color: iconColor, size: 16),
            const SizedBox(width: 8),
            Text(title, style: Theme.of(context).textTheme.titleMedium),
            const SizedBox(width: 8),
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 7, vertical: 2),
              decoration: BoxDecoration(
                color: iconColor.withOpacity(0.12),
                borderRadius: BorderRadius.circular(5),
              ),
              child: Text(
                '${skills.length}',
                style: GoogleFonts.fraunces(
                    fontSize: 11,
                    fontWeight: FontWeight.w700,
                    color: iconColor),
              ),
            ),
          ],
        ),
        if (subtitle != null) ...[
          const SizedBox(height: 4),
          Text(subtitle, style: Theme.of(context).textTheme.bodyMedium),
        ],
        const SizedBox(height: 10),
        Wrap(
          spacing: 7,
          runSpacing: 7,
          children: skills.map((s) => SkillChip(label: s, type: type)).toList(),
        ),
      ],
    );
  }

  // ── Missing skills section with add-skill input ───────────────────────────

  Widget _buildMissingSkillsSection(BuildContext context) {
    final c = AppColors.of(context);
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          children: [
            const Icon(Icons.radio_button_unchecked_rounded,
                color: AppTheme.red500, size: 16),
            const SizedBox(width: 8),
            Text('Missing Skills',
                style: Theme.of(context).textTheme.titleMedium),
            const SizedBox(width: 8),
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 7, vertical: 2),
              decoration: BoxDecoration(
                color: AppTheme.red500.withOpacity(0.12),
                borderRadius: BorderRadius.circular(5),
              ),
              child: Text(
                '${_missingSkills.length}',
                style: GoogleFonts.fraunces(
                    fontSize: 11,
                    fontWeight: FontWeight.w700,
                    color: AppTheme.red500),
              ),
            ),
            if (_savingSkills) ...[
              const SizedBox(width: 8),
              SizedBox(
                width: 11,
                height: 11,
                child: CircularProgressIndicator(
                    strokeWidth: 1.6, color: c.textTertiary),
              ),
            ],
          ],
        ),
        const SizedBox(height: 4),
        Text('Add skills your resume is missing to track your progress',
            style: Theme.of(context).textTheme.bodyMedium),
        const SizedBox(height: 12),

        // ── Add skill text field ──────────────────────────────────────
        Row(
          children: [
            Expanded(
              child: TextFormField(
                controller: _addSkillCtrl,
                style: GoogleFonts.workSans(fontSize: 13, color: c.textPrimary),
                decoration: InputDecoration(
                  hintText: 'e.g. Docker, Machine Learning, SQL…',
                  hintStyle:
                      GoogleFonts.workSans(fontSize: 13, color: c.textTertiary),
                  prefixIcon: Icon(Icons.add_circle_outline_rounded,
                      color: c.textTertiary, size: 18),
                  contentPadding:
                      const EdgeInsets.symmetric(vertical: 12, horizontal: 14),
                ),
                onFieldSubmitted: (_) => _addSkill(),
                textInputAction: TextInputAction.done,
              ),
            ),
            const SizedBox(width: 8),
            ElevatedButton(
              onPressed: _addSkill,
              style: ElevatedButton.styleFrom(
                padding:
                    const EdgeInsets.symmetric(horizontal: 16, vertical: 14),
                shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(10)),
              ),
              child: Text('Add',
                  style: GoogleFonts.fraunces(
                      fontSize: 13, fontWeight: FontWeight.w700)),
            ),
          ],
        ),
        const SizedBox(height: 12),

        // ── Missing skill chips (removable) ───────────────────────────
        if (_missingSkills.isEmpty)
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 12),
            decoration: BoxDecoration(
              color: c.navy800,
              borderRadius: BorderRadius.circular(10),
              border: Border.all(color: c.cardBorder),
            ),
            child: Row(children: [
              Icon(Icons.check_circle_outline_rounded,
                  color: AppTheme.green500, size: 15),
              const SizedBox(width: 8),
              Text('No missing skills — great job!',
                  style: GoogleFonts.workSans(
                      fontSize: 13, color: c.textSecondary)),
            ]),
          )
        else
          Wrap(
            spacing: 7,
            runSpacing: 7,
            children: _missingSkills
                .map((s) => _removableSkillChip(context, s))
                .toList(),
          ),
      ],
    );
  }

  Widget _removableSkillChip(BuildContext context, String skill) {
    final c = AppColors.of(context);
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
      decoration: BoxDecoration(
        color: AppTheme.red500.withOpacity(0.08),
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: AppTheme.red500.withOpacity(0.3)),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Text(
            skill,
            style: GoogleFonts.workSans(
                fontSize: 12,
                color: AppTheme.red400,
                fontWeight: FontWeight.w500),
          ),
          const SizedBox(width: 6),
          GestureDetector(
            onTap: () => _removeSkill(skill),
            child: Icon(Icons.close_rounded, size: 14, color: AppTheme.red400),
          ),
        ],
      ),
    );
  }

  Widget _buildAllSkillsSection(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const LabeledDivider(label: 'All discovered skills'),
        const SizedBox(height: 14),
        Wrap(
          spacing: 7,
          runSpacing: 7,
          children: widget.record.skills
              .map((s) => SkillChip(label: s, type: SkillChipType.neutral))
              .toList(),
        ),
      ],
    );
  }

  Widget _buildCertSection(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const LabeledDivider(label: 'Certifications'),
        const SizedBox(height: 14),
        Wrap(
          spacing: 7,
          runSpacing: 7,
          children: widget.record.certs
              .map((c) => SkillChip(label: c, type: SkillChipType.neutral))
              .toList(),
        ),
      ],
    );
  }

  Widget _buildMetaCard(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: AppColors.of(context).navy800,
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: AppColors.of(context).cardBorder),
      ),
      child: Column(
        children: [
          _metaRow(context, 'Resume file', widget.record.fileName),
          const Divider(height: 16),
          _metaRow(
              context, 'Upload mode', widget.record.uploadMode.toUpperCase()),
          const Divider(height: 16),
          _metaRow(context, 'Industry',
              widget.record.industry.isNotEmpty ? widget.record.industry : '—'),
          const Divider(height: 16),
          _metaRow(
              context,
              'Scanned at',
              DateFormat('MMM d, yyyy  hh:mm a')
                  .format(widget.record.scannedAt)),
        ],
      ),
    );
  }

  Widget _metaRow(BuildContext context, String key, String value) {
    return Row(
      mainAxisAlignment: MainAxisAlignment.spaceBetween,
      children: [
        Text(key, style: Theme.of(context).textTheme.bodyMedium),
        Flexible(
          child: Text(
            value,
            style: GoogleFonts.workSans(
                fontSize: 12,
                color: AppColors.of(context).textSecondary,
                fontWeight: FontWeight.w500),
            textAlign: TextAlign.end,
            overflow: TextOverflow.ellipsis,
          ),
        ),
      ],
    );
  }
}
