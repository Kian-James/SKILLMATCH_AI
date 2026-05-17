// lib/screens/results_screen.dart

import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:intl/intl.dart';
import '../models/models.dart';
import '../theme.dart';
import '../widgets/common_widgets.dart';

class ResultsScreen extends StatelessWidget {
  final ResumeScanRecord record;
  final ResumeUser user;
  final bool fromCache;

  const ResultsScreen({
    super.key,
    required this.record,
    required this.user,
    required this.fromCache,
  });

  double get _accuracyValue {
    final clean = record.nlpAccuracy.replaceAll('%', '').trim();
    return double.tryParse(clean) ?? 0;
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text(
          record.targetJobTitle,
          overflow: TextOverflow.ellipsis,
        ),
        actions: [
          if (fromCache)
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
                record.matchedSkills, SkillChipType.matched,
                icon: Icons.check_circle_outline_rounded,
                iconColor: AppTheme.green500),
            const SizedBox(height: 20),
            _buildSkillsSection(context, 'Missing Skills',
                record.missingSkills, SkillChipType.missing,
                icon: Icons.radio_button_unchecked_rounded,
                iconColor: AppTheme.red500,
                subtitle: 'Add these to your resume to improve your match'),
            const SizedBox(height: 20),
            _buildAllSkillsSection(context),
            const SizedBox(height: 20),
            if (record.certCount > 0) _buildCertSection(context),
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
        color: AppTheme.navy800,
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
                Text(record.targetJobTitle,
                    style: Theme.of(context).textTheme.titleLarge),
                const SizedBox(height: 3),
                Text(record.industry.isNotEmpty ? record.industry : 'Industry',
                    style: Theme.of(context).textTheme.bodyMedium),
                const SizedBox(height: 6),
                Text(record.fileName,
                    style: GoogleFonts.inter(
                        fontSize: 11,
                        color: AppTheme.textTertiary,
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
          value: '${record.skillCount}',
          valueColor: AppTheme.textPrimary,
          icon: Icons.psychology_outlined,
        ),
        StatCard(
          label: 'Matched',
          value: '${record.matchedCount}',
          valueColor: AppTheme.green500,
          icon: Icons.check_circle_outline_rounded,
        ),
        StatCard(
          label: 'Missing',
          value: '${record.missingCount}',
          valueColor: AppTheme.red500,
          icon: Icons.remove_circle_outline_rounded,
        ),
        StatCard(
          label: 'Certifications',
          value: '${record.certCount}',
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
        color: AppTheme.navy800,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: AppTheme.cardBorder),
      ),
      child: Row(
        children: [
          AccuracyRing(accuracy: record.nlpAccuracy, size: 80),
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
                    backgroundColor: AppTheme.navy600,
                    valueColor:
                        AlwaysStoppedAnimation<Color>(_accuracyColor),
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
              padding:
                  const EdgeInsets.symmetric(horizontal: 7, vertical: 2),
              decoration: BoxDecoration(
                color: iconColor.withOpacity(0.12),
                borderRadius: BorderRadius.circular(5),
              ),
              child: Text(
                '${skills.length}',
                style: GoogleFonts.spaceGrotesk(
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

  Widget _buildAllSkillsSection(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const LabeledDivider(label: 'All discovered skills'),
        const SizedBox(height: 14),
        Wrap(
          spacing: 7,
          runSpacing: 7,
          children: record.skills
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
          children: record.certs
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
        color: AppTheme.navy800,
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: AppTheme.cardBorder),
      ),
      child: Column(
        children: [
          _metaRow(context, 'Resume file', record.fileName),
          const Divider(height: 16),
          _metaRow(context, 'Upload mode', record.uploadMode.toUpperCase()),
          const Divider(height: 16),
          _metaRow(context, 'Industry',
              record.industry.isNotEmpty ? record.industry : '—'),
          const Divider(height: 16),
          _metaRow(context, 'Scanned at',
              DateFormat('MMM d, yyyy  hh:mm a').format(record.scannedAt)),
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
            style: GoogleFonts.inter(
                fontSize: 12,
                color: AppTheme.textSecondary,
                fontWeight: FontWeight.w500),
            textAlign: TextAlign.end,
            overflow: TextOverflow.ellipsis,
          ),
        ),
      ],
    );
  }
}
