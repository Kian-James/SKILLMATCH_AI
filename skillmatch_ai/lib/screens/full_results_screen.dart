// lib/screens/full_results_screen.dart

import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:intl/intl.dart';

import '../models/models.dart';
import '../theme.dart';
import '../widgets/common_widgets.dart';

class FullResultsScreen extends StatelessWidget {
  final ResumeScanRecord record;
  final ResumeUser user;
  final bool fromCache;

  const FullResultsScreen({
    super.key,
    required this.record,
    required this.user,
    required this.fromCache,
  });

  @override
  Widget build(BuildContext context) {
    final c = AppColors.of(context);
    return Scaffold(
      appBar: AppBar(
        title: const Text('Full Analysis'),
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
            _buildScanMeta(context),
            const SizedBox(height: 20),

            // ── Step 1: How we read the resume ───────────────────────────
            _stepHeader(context, '1', 'Resume Reading'),
            const SizedBox(height: 12),
            _buildStep1(context),
            const SizedBox(height: 24),

            // ── Step 2: Skills found ──────────────────────────────────────
            _stepHeader(context, '2', 'Skills Found'),
            const SizedBox(height: 12),
            _buildSkillChips(context),
            const SizedBox(height: 24),

            // ── Step 2B: Certifications ───────────────────────────────────
            if (record.certCount > 0) ...[
              _stepHeader(context, '2B', 'Certifications'),
              const SizedBox(height: 12),
              _buildCertChips(context),
              const SizedBox(height: 24),
            ],

            // ── Step 3: Career matches ────────────────────────────────────
            _stepHeader(context, '3', 'Best Career Matches'),
            const SizedBox(height: 4),
            Text(
              'Ranked by how well your skills align with each role',
              style: Theme.of(context).textTheme.bodyMedium,
            ),
            const SizedBox(height: 14),
            _buildTop5(context),
            const SizedBox(height: 24),

            // ── Step 4: System check ──────────────────────────────────────
            _stepHeader(context, '4', 'Analysis Quality Check'),
            const SizedBox(height: 12),
            _buildQualityCheck(context),
            const SizedBox(height: 32),
          ],
        ),
      ),
    );
  }

  // ── Scan meta header ──────────────────────────────────────────────────────

  Widget _buildScanMeta(BuildContext context) {
    final c = AppColors.of(context);
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: c.navy800,
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: c.brand.withOpacity(0.25)),
      ),
      child: Row(children: [
        Container(
          width: 44,
          height: 44,
          decoration: BoxDecoration(
            color: c.brand.withOpacity(0.1),
            borderRadius: BorderRadius.circular(12),
          ),
          child: Icon(Icons.grid_view_rounded, color: c.brand, size: 20),
        ),
        const SizedBox(width: 12),
        Expanded(
            child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(record.fileName,
                style: Theme.of(context).textTheme.titleMedium,
                overflow: TextOverflow.ellipsis),
            const SizedBox(height: 3),
            Text(
              DateFormat('MMM d, yyyy  hh:mm a').format(record.scannedAt),
              style: GoogleFonts.inter(fontSize: 11, color: c.textTertiary),
            ),
          ],
        )),
        _modeBadge(context, record.uploadMode),
      ]),
    );
  }

  Widget _modeBadge(BuildContext context, String mode) {
    final c = AppColors.of(context);
    final label = switch (mode) {
      'pdf' => 'PDF',
      'image' => 'Image',
      'scan' => 'Camera',
      _ => mode.toUpperCase(),
    };
    final icon = switch (mode) {
      'pdf' => Icons.picture_as_pdf_outlined,
      'image' => Icons.image_outlined,
      'scan' => Icons.document_scanner_outlined,
      _ => Icons.file_present_outlined,
    };
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
      decoration: BoxDecoration(
        color: c.navy700,
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: c.cardBorder),
      ),
      child: Row(mainAxisSize: MainAxisSize.min, children: [
        Icon(icon, size: 13, color: c.textTertiary),
        const SizedBox(width: 5),
        Text(label,
            style: GoogleFonts.inter(
                fontSize: 11,
                color: c.textTertiary,
                fontWeight: FontWeight.w600)),
      ]),
    );
  }

  // ── Step header ──────────────────────────────────────────────────────────

  Widget _stepHeader(BuildContext context, String number, String title) {
    final c = AppColors.of(context);
    return Row(children: [
      Container(
        width: 28,
        height: 28,
        decoration: BoxDecoration(
          color: c.brand.withOpacity(0.12),
          borderRadius: BorderRadius.circular(8),
          border: Border.all(color: c.brand.withOpacity(0.3)),
        ),
        child: Center(
          child: Text(number,
              style: GoogleFonts.spaceGrotesk(
                  fontSize: 11,
                  fontWeight: FontWeight.w700,
                  color: c.brandSoft)),
        ),
      ),
      const SizedBox(width: 10),
      Text(title, style: Theme.of(context).textTheme.titleMedium),
    ]);
  }

  // ── Step 1: Resume reading ────────────────────────────────────────────────

  Widget _buildStep1(BuildContext context) {
    final c = AppColors.of(context);
    final isCamera = record.uploadMode == 'scan';

    // Human-friendly source label
    final sourceLabel = switch (record.uploadMode) {
      'pdf' => 'PDF file',
      'image' => 'Image file',
      'scan' => 'Camera scan',
      _ => record.uploadMode,
    };

    // Human-friendly reading quality label
    final qualityLabel = switch (record.uploadMode) {
      'pdf' => 'High quality — text extracted directly',
      'scan' => 'Good quality — text recognised from camera',
      _ => 'Good quality — text recognised from image',
    };

    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: c.navy800,
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: c.cardBorder),
      ),
      child: Column(children: [
        _metaRow(context, 'Source', sourceLabel),
        Divider(height: 16, color: c.divider),
        _metaRow(context, 'Found',
            '${record.skillCount} skills  •  ${record.certCount} certificates'),
        Divider(height: 16, color: c.divider),
        Row(mainAxisAlignment: MainAxisAlignment.spaceBetween, children: [
          Text('Reading quality',
              style: Theme.of(context).textTheme.bodyMedium),
          Text(
            qualityLabel,
            style: GoogleFonts.inter(
                fontSize: 12, color: c.brandSoft, fontWeight: FontWeight.w500),
          ),
        ]),
      ]),
    );
  }

  // ── Step 2: Skills ────────────────────────────────────────────────────────

  Widget _buildSkillChips(BuildContext context) {
    final c = AppColors.of(context);
    if (record.skills.isEmpty) {
      return _emptyBox(context, 'No skills were found in your resume');
    }
    return Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
      Row(children: [
        Icon(Icons.psychology_outlined, color: c.brand, size: 15),
        const SizedBox(width: 7),
        Text('${record.skillCount} skills found',
            style: GoogleFonts.inter(fontSize: 12, color: c.textSecondary)),
      ]),
      const SizedBox(height: 10),
      Wrap(
          spacing: 7,
          runSpacing: 7,
          children: record.skills
              .map((s) => SkillChip(label: s, type: SkillChipType.neutral))
              .toList()),
    ]);
  }

  // ── Step 2B: Certs ────────────────────────────────────────────────────────

  Widget _buildCertChips(BuildContext context) {
    final c = AppColors.of(context);
    return Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
      Row(children: [
        Icon(Icons.verified_outlined, color: AppTheme.amber500, size: 15),
        const SizedBox(width: 7),
        Text('${record.certCount} certifications found',
            style: GoogleFonts.inter(fontSize: 12, color: c.textSecondary)),
      ]),
      const SizedBox(height: 10),
      Wrap(
          spacing: 7,
          runSpacing: 7,
          children: record.certs
              .map(
                  (cert) => SkillChip(label: cert, type: SkillChipType.neutral))
              .toList()),
    ]);
  }

  // ── Step 3: Top 5 ─────────────────────────────────────────────────────────

  Widget _buildTop5(BuildContext context) {
    final matches = record.top5Matches;
    if (matches.isEmpty) {
      return _emptyBox(context, 'No career matches available');
    }
    return Column(
      children: matches.asMap().entries.map((e) {
        return Padding(
          padding: EdgeInsets.only(bottom: e.key < matches.length - 1 ? 12 : 0),
          child: _Top5Card(match: e.value, isTop: e.key == 0),
        );
      }).toList(),
    );
  }

  // ── Step 4: Quality check (was FAISS integrity) ───────────────────────────

  Widget _buildQualityCheck(BuildContext context) {
    final c = AppColors.of(context);
    final score = record.faissScore;
    final isPassing = score == '100.0%' || score == '100%';
    final color = isPassing ? AppTheme.green500 : AppTheme.red500;

    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: c.navy800,
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: color.withOpacity(0.25)),
      ),
      child: Row(children: [
        Container(
          width: 52,
          height: 52,
          decoration: BoxDecoration(
            color: color.withOpacity(0.1),
            shape: BoxShape.circle,
          ),
          child: Icon(
            isPassing ? Icons.verified_rounded : Icons.warning_amber_rounded,
            color: color,
            size: 24,
          ),
        ),
        const SizedBox(width: 16),
        Expanded(
            child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text('Analysis complete',
                style: Theme.of(context).textTheme.titleMedium),
            const SizedBox(height: 4),
            Text(
              isPassing
                  ? 'Your results were stored and verified successfully'
                  : 'Something went wrong saving your results — try again',
              style: GoogleFonts.inter(fontSize: 12, color: c.textSecondary),
            ),
          ],
        )),
        Text(isPassing ? '✓' : '!',
            style: GoogleFonts.spaceGrotesk(
                fontSize: 22, fontWeight: FontWeight.w700, color: color)),
      ]),
    );
  }

  // ── Shared helpers ────────────────────────────────────────────────────────

  Widget _metaRow(BuildContext context, String key, String value) {
    final c = AppColors.of(context);
    return Row(mainAxisAlignment: MainAxisAlignment.spaceBetween, children: [
      Text(key, style: Theme.of(context).textTheme.bodyMedium),
      Text(value,
          style: GoogleFonts.inter(
              fontSize: 12,
              color: c.textSecondary,
              fontWeight: FontWeight.w500)),
    ]);
  }

  Widget _emptyBox(BuildContext context, String msg) {
    final c = AppColors.of(context);
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: c.navy800,
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: c.cardBorder),
      ),
      child: Text(msg, style: Theme.of(context).textTheme.bodyMedium),
    );
  }
}

// ── Career match expandable card ──────────────────────────────────────────────

class _Top5Card extends StatefulWidget {
  final Top5Match match;
  final bool isTop;

  const _Top5Card({required this.match, required this.isTop});

  @override
  State<_Top5Card> createState() => _Top5CardState();
}

class _Top5CardState extends State<_Top5Card>
    with SingleTickerProviderStateMixin {
  bool _expanded = false;
  late AnimationController _ctrl;
  late Animation<double> _expandAnim;

  @override
  void initState() {
    super.initState();
    _expanded = widget.isTop; // auto-expand the #1 match
    _ctrl = AnimationController(
        vsync: this, duration: const Duration(milliseconds: 250));
    _expandAnim = CurvedAnimation(parent: _ctrl, curve: Curves.easeInOut);
    if (_expanded) _ctrl.value = 1.0;
  }

  @override
  void dispose() {
    _ctrl.dispose();
    super.dispose();
  }

  void _toggle() {
    setState(() => _expanded = !_expanded);
    _expanded ? _ctrl.forward() : _ctrl.reverse();
  }

  Color get _rankColor => switch (widget.match.rank) {
        1 => AppTheme.teal500,
        2 => AppTheme.green500,
        3 => AppTheme.amber500,
        _ => AppTheme.textTertiary,
      };

  @override
  Widget build(BuildContext context) {
    final m = widget.match;
    final c = AppColors.of(context);

    return Container(
      decoration: BoxDecoration(
        color: c.navy800,
        borderRadius: BorderRadius.circular(14),
        border: Border.all(
          color: widget.isTop ? c.brand.withOpacity(0.35) : c.cardBorder,
          width: widget.isTop ? 1.5 : 1,
        ),
      ),
      child: Column(children: [
        // ── Header row ────────────────────────────────────────────────
        GestureDetector(
          onTap: _toggle,
          behavior: HitTestBehavior.opaque,
          child: Padding(
            padding: const EdgeInsets.all(14),
            child: Row(children: [
              // Rank badge
              Container(
                width: 32,
                height: 32,
                decoration: BoxDecoration(
                  color: _rankColor.withOpacity(0.12),
                  borderRadius: BorderRadius.circular(8),
                ),
                child: Center(
                  child: Text('#${m.rank}',
                      style: GoogleFonts.spaceGrotesk(
                          fontSize: 12,
                          fontWeight: FontWeight.w700,
                          color: _rankColor)),
                ),
              ),
              const SizedBox(width: 12),

              // Title + skill requirement count
              Expanded(
                  child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(m.title,
                      style: Theme.of(context).textTheme.titleMedium,
                      overflow: TextOverflow.ellipsis),
                  const SizedBox(height: 2),
                  Text(
                    m.industry.isNotEmpty
                        ? m.industry
                        : '${m.groundTruthCount} skills required',
                    style:
                        GoogleFonts.inter(fontSize: 11, color: c.textTertiary),
                  ),
                ],
              )),
              const SizedBox(width: 10),

              // Match score ring
              AccuracyRing(accuracy: m.nlpAccuracy, size: 48),
              const SizedBox(width: 8),

              // Expand chevron
              AnimatedRotation(
                turns: _expanded ? 0.5 : 0,
                duration: const Duration(milliseconds: 250),
                child: Icon(Icons.keyboard_arrow_down_rounded,
                    color: c.textTertiary, size: 20),
              ),
            ]),
          ),
        ),

        // ── Expanded body ──────────────────────────────────────────────
        SizeTransition(
          sizeFactor: _expandAnim,
          child: Column(children: [
            Divider(height: 1, color: c.divider, indent: 14, endIndent: 14),
            Padding(
              padding: const EdgeInsets.fromLTRB(14, 14, 14, 16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  // ── Why this role fits ────────────────────────────────
                  if (m.fitReason.isNotEmpty) ...[
                    Container(
                      width: double.infinity,
                      padding: const EdgeInsets.symmetric(
                          horizontal: 12, vertical: 10),
                      decoration: BoxDecoration(
                        color: c.brand.withOpacity(0.07),
                        borderRadius: BorderRadius.circular(10),
                        border: Border.all(color: c.brand.withOpacity(0.2)),
                      ),
                      child: Row(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Icon(Icons.lightbulb_outline_rounded,
                              color: c.brandSoft, size: 14),
                          const SizedBox(width: 8),
                          Expanded(
                            child: Text(
                              m.fitReason,
                              style: GoogleFonts.inter(
                                fontSize: 12,
                                color: c.brandSoft,
                                height: 1.5,
                              ),
                            ),
                          ),
                        ],
                      ),
                    ),
                    const SizedBox(height: 14),
                  ],

                  // ── Skills you already have ───────────────────────────
                  if (m.matchedSkills.isNotEmpty) ...[
                    _sectionLabel(
                        context,
                        Icons.check_circle_outline_rounded,
                        AppTheme.green500,
                        'Skills you already have (${m.matchedSkills.length})'),
                    const SizedBox(height: 8),
                    Wrap(
                        spacing: 6,
                        runSpacing: 6,
                        children: m.matchedSkills
                            .map((s) => SkillChip(
                                label: s, type: SkillChipType.matched))
                            .toList()),
                    const SizedBox(height: 14),
                  ],

                  // ── Skills to develop ─────────────────────────────────
                  if (m.unmatchedSkills.isNotEmpty) ...[
                    _sectionLabel(
                        context,
                        Icons.radio_button_unchecked_rounded,
                        AppTheme.red500,
                        'Skills to develop (${m.unmatchedSkills.length})'),
                    const SizedBox(height: 8),
                    Wrap(
                        spacing: 6,
                        runSpacing: 6,
                        children: m.unmatchedSkills
                            .map((s) => SkillChip(
                                label: s, type: SkillChipType.missing))
                            .toList()),
                  ],
                ],
              ),
            ),
          ]),
        ),
      ]),
    );
  }

  Widget _sectionLabel(
      BuildContext context, IconData icon, Color color, String label) {
    return Row(children: [
      Icon(icon, color: color, size: 14),
      const SizedBox(width: 6),
      Text(label,
          style: GoogleFonts.spaceGrotesk(
              fontSize: 12, fontWeight: FontWeight.w600, color: color)),
    ]);
  }
}
