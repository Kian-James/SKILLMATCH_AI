// lib/screens/full_results_screen.dart

import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:intl/intl.dart';

import '../models/models.dart';
import '../services/firestore_service.dart';
import '../theme.dart';
import '../utils/responsive.dart';
import '../widgets/common_widgets.dart';

class FullResultsScreen extends StatefulWidget {
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
  State<FullResultsScreen> createState() => _FullResultsScreenState();
}

class _FullResultsScreenState extends State<FullResultsScreen> {
  late List<String> _mySkills;
  late List<String> _myCerts;
  final _addSkillCtrl = TextEditingController();
  final _addCertCtrl = TextEditingController();
  final _firestore = FirestoreService();

  @override
  void initState() {
    super.initState();
    _mySkills = List<String>.from(widget.record.skills);
    _myCerts = List<String>.from(widget.record.certs);
  }

  @override
  void dispose() {
    _addSkillCtrl.dispose();
    _addCertCtrl.dispose();
    super.dispose();
  }

  // ── Skill management ──────────────────────────────────────────────────────

  void _addSkill(String skill) {
    final trimmed = skill.trim();
    if (trimmed.isEmpty) return;
    if (_mySkills.map((s) => s.toLowerCase()).contains(trimmed.toLowerCase()))
      return;
    setState(() => _mySkills.add(trimmed));
    _persistSkills();
  }

  void _removeSkill(String skill) {
    setState(() => _mySkills.remove(skill));
    _persistSkills();
  }

  void _addSkillFromField() {
    _addSkill(_addSkillCtrl.text);
    _addSkillCtrl.clear();
  }

  void _persistSkills() {
    final docId = widget.record.id;
    if (docId.isEmpty) {
      debugPrint(
          '[FullResultsScreen] Cannot save skills — record.id is empty.');
      _showSaveWarning('skills');
      return;
    }
    _firestore.updateSkills(docId, List<String>.from(_mySkills)).then((_) {
      debugPrint(
          '[FullResultsScreen] Skills saved to Firestore ($docId): $_mySkills');
    }).catchError((e) {
      debugPrint('[FullResultsScreen] FAILED to save skills: $e');
      if (mounted) _showSaveError('skills', e);
    });
  }

  // ── Cert management ───────────────────────────────────────────────────────

  void _addCert(String cert) {
    final trimmed = cert.trim();
    if (trimmed.isEmpty) return;
    if (_myCerts.map((s) => s.toLowerCase()).contains(trimmed.toLowerCase()))
      return;
    setState(() => _myCerts.add(trimmed));
    _persistCerts();
  }

  void _removeCert(String cert) {
    setState(() => _myCerts.remove(cert));
    _persistCerts();
  }

  void _addCertFromField() {
    _addCert(_addCertCtrl.text);
    _addCertCtrl.clear();
  }

  void _persistCerts() {
    final docId = widget.record.id;
    if (docId.isEmpty) {
      debugPrint('[FullResultsScreen] Cannot save certs — record.id is empty.');
      _showSaveWarning('certifications');
      return;
    }
    _firestore.updateCerts(docId, List<String>.from(_myCerts)).then((_) {
      debugPrint(
          '[FullResultsScreen] Certs saved to Firestore ($docId): $_myCerts');
    }).catchError((e) {
      debugPrint('[FullResultsScreen] FAILED to save certs: $e');
      if (mounted) _showSaveError('certifications', e);
    });
  }

  void _showSaveWarning(String what) {
    if (!mounted) return;
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Text(
            'Could not save $what — this scan has no record ID yet. Try again after the scan finishes saving.'),
        backgroundColor: AppTheme.amber500,
      ),
    );
  }

  void _showSaveError(String what, Object error) {
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Text('Failed to save $what: $error'),
        backgroundColor: AppTheme.red500,
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final c = AppColors.of(context);
    return Scaffold(
      appBar: AppBar(
        title: const Text('Full Analysis'),
        actions: [
          if (widget.fromCache)
            Container(
              margin: const EdgeInsets.only(right: 16),
              child: const TealBadge(text: 'Cached'),
            ),
        ],
      ),
      body: ResponsiveBody(
        maxWidth: 800,
        child: SingleChildScrollView(
        padding: EdgeInsets.symmetric(
            horizontal: responsiveGutter(context, 20), vertical: 16),
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
            _stepHeader(context, '2B', 'Certifications'),
            const SizedBox(height: 12),
            _buildCertChips(context),
            const SizedBox(height: 24),

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
        border: Border.all(color: c.cardBorder, width: 2.5),
        boxShadow: AppTheme.hardShadow(c.cardBorder, dx: 4, dy: 4),
      ),
      child: Row(children: [
        Container(
          width: 44,
          height: 44,
          decoration: BoxDecoration(
            color: c.brand,
            borderRadius: BorderRadius.circular(12),
            border: Border.all(color: c.cardBorder, width: 1.75),
          ),
          child: const Icon(Icons.grid_view_rounded, color: AppTheme.ink, size: 20),
        ),
        const SizedBox(width: 12),
        Expanded(
            child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(widget.record.fileName,
                style: Theme.of(context).textTheme.titleMedium,
                overflow: TextOverflow.ellipsis),
            const SizedBox(height: 3),
            Text(
              DateFormat('MMM d, yyyy  hh:mm a')
                  .format(widget.record.scannedAt),
              style: GoogleFonts.workSans(fontSize: 11, color: c.textTertiary),
            ),
          ],
        )),
        _modeBadge(context, widget.record.uploadMode),
      ]),
    );
  }

  Widget _modeBadge(BuildContext context, String mode) {
    final c = AppColors.of(context);
    final label = switch (mode) {
      'pdf' => 'PDF',
      'image' => 'Image',
      _ => mode.toUpperCase(),
    };
    final icon = switch (mode) {
      'pdf' => Icons.picture_as_pdf_outlined,
      'image' => Icons.image_outlined,
      _ => Icons.file_present_outlined,
    };
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
      decoration: BoxDecoration(
        color: c.navy700,
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: c.cardBorder, width: 2.5),
      ),
      child: Row(mainAxisSize: MainAxisSize.min, children: [
        Icon(icon, size: 13, color: c.textTertiary),
        const SizedBox(width: 5),
        Text(label,
            style: GoogleFonts.workSans(
                fontSize: 11,
                color: c.textTertiary,
                fontWeight: FontWeight.w600)),
      ]),
    );
  }

  // ── Step header ───────────────────────────────────────────────────────────

  Widget _stepHeader(BuildContext context, String number, String title) {
    final c = AppColors.of(context);
    return Row(children: [
      Container(
        width: 28,
        height: 28,
        decoration: BoxDecoration(
          color: c.brand,
          borderRadius: BorderRadius.circular(8),
          border: Border.all(color: c.cardBorder, width: 2),
        ),
        child: Center(
          child: Text(number,
              style: GoogleFonts.spaceGrotesk(
                  fontSize: 11,
                  fontWeight: FontWeight.w700,
                  color: AppTheme.ink)),
        ),
      ),
      const SizedBox(width: 10),
      Text(title, style: Theme.of(context).textTheme.titleMedium),
    ]);
  }

  // ── Step 1: Resume reading ────────────────────────────────────────────────

  Widget _buildStep1(BuildContext context) {
    final c = AppColors.of(context);

    final sourceLabel = switch (widget.record.uploadMode) {
      'pdf' => 'PDF file',
      'image' => 'Image file',
      _ => widget.record.uploadMode,
    };

    final qualityLabel = switch (widget.record.uploadMode) {
      'pdf' => 'High quality — text extracted directly',
      _ => 'Good quality — text recognised from image',
    };

    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: c.navy800,
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: c.cardBorder, width: 2.5),
        boxShadow: AppTheme.hardShadow(c.cardBorder, dx: 4, dy: 4),
      ),
      child: Column(children: [
        _metaRow(context, 'Source', sourceLabel),
        Divider(height: 16, color: c.divider),
        _metaRow(context, 'Found',
            '${_mySkills.length} skills  •  ${_myCerts.length} certificates'),
        Divider(height: 16, color: c.divider),
        Row(mainAxisAlignment: MainAxisAlignment.spaceBetween, children: [
          Text('Reading quality',
              style: Theme.of(context).textTheme.bodyMedium),
          Text(
            qualityLabel,
            style: GoogleFonts.workSans(
                fontSize: 12, color: c.brandSoft, fontWeight: FontWeight.w500),
          ),
        ]),
      ]),
    );
  }

  // ── Step 2: Skills (with add input) ──────────────────────────────────────

  Widget _buildSkillChips(BuildContext context) {
    final c = AppColors.of(context);
    return Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
      // ── Add skill input ───────────────────────────────────────────────
      _addItemRow(
        context,
        controller: _addSkillCtrl,
        hint: 'Add a skill to your profile…',
        onAdd: _addSkillFromField,
      ),
      const SizedBox(height: 12),

      // ── Skill chips ───────────────────────────────────────────────────
      if (_mySkills.isEmpty)
        _emptyBox(context, 'No skills yet — add one above')
      else ...[
        Row(children: [
          Icon(Icons.psychology_outlined, color: c.brand, size: 15),
          const SizedBox(width: 7),
          Text('${_mySkills.length} skills found',
              style:
                  GoogleFonts.workSans(fontSize: 12, color: c.textSecondary)),
        ]),
        const SizedBox(height: 10),
        Wrap(
            spacing: 7,
            runSpacing: 7,
            children: _mySkills
                .map((s) => _removableNeutralChip(context, s,
                    onRemove: () => _removeSkill(s)))
                .toList()),
      ],
    ]);
  }

  // ── Step 2B: Certs (with add input) ──────────────────────────────────────

  Widget _buildCertChips(BuildContext context) {
    final c = AppColors.of(context);
    return Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
      // ── Add cert input ────────────────────────────────────────────────
      _addItemRow(
        context,
        controller: _addCertCtrl,
        hint: 'Add a certification to your profile…',
        onAdd: _addCertFromField,
      ),
      const SizedBox(height: 12),

      // ── Cert chips ────────────────────────────────────────────────────
      if (_myCerts.isEmpty)
        _emptyBox(context, 'No certifications yet — add one above')
      else ...[
        Row(children: [
          Icon(Icons.verified_outlined, color: AppTheme.amber500, size: 15),
          const SizedBox(width: 7),
          Text('${_myCerts.length} certifications found',
              style:
                  GoogleFonts.workSans(fontSize: 12, color: c.textSecondary)),
        ]),
        const SizedBox(height: 10),
        Wrap(
            spacing: 7,
            runSpacing: 7,
            children: _myCerts
                .map((cert) => _removableNeutralChip(context, cert,
                    onRemove: () => _removeCert(cert),
                    color: AppTheme.amber500))
                .toList()),
      ],
    ]);
  }

  // ── Step 3: Top 5 ─────────────────────────────────────────────────────────

  Widget _buildTop5(BuildContext context) {
    final matches = widget.record.top5Matches;
    if (matches.isEmpty) {
      return _emptyBox(context, 'No career matches available');
    }
    return Column(
      children: matches.asMap().entries.map((e) {
        return Padding(
          padding: EdgeInsets.only(bottom: e.key < matches.length - 1 ? 12 : 0),
          child: _Top5Card(
            match: e.value,
            isTop: e.key == 0,
            mySkills: _mySkills,
            onAddSkill: _addSkill,
            onRemoveSkill: _removeSkill,
          ),
        );
      }).toList(),
    );
  }

  // ── Step 4: Quality check ─────────────────────────────────────────────────

  Widget _buildQualityCheck(BuildContext context) {
    final c = AppColors.of(context);
    final scoreValue =
        double.tryParse(widget.record.faissScore.replaceAll('%', '').trim()) ??
            0;
    final isPassing = scoreValue >= 99.5;
    final color = isPassing ? AppTheme.green500 : AppTheme.red500;

    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: c.navy800,
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: c.cardBorder, width: 2.5),
        boxShadow: AppTheme.hardShadow(c.cardBorder, dx: 4, dy: 4),
      ),
      child: Row(children: [
        Container(
          width: 52,
          height: 52,
          decoration: BoxDecoration(
            color: color,
            borderRadius: BorderRadius.circular(14),
            border: Border.all(color: c.cardBorder, width: 1.75),
          ),
          child: Icon(
            isPassing ? Icons.verified_rounded : Icons.warning_amber_rounded,
            color: AppTheme.ink,
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
              style: GoogleFonts.workSans(fontSize: 12, color: c.textSecondary),
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

  /// Reusable text field + Add button row
  Widget _addItemRow(
    BuildContext context, {
    required TextEditingController controller,
    required String hint,
    required VoidCallback onAdd,
  }) {
    final c = AppColors.of(context);
    return Row(children: [
      Expanded(
        child: TextFormField(
          controller: controller,
          style: GoogleFonts.workSans(fontSize: 13, color: c.textPrimary),
          decoration: InputDecoration(
            hintText: hint,
            hintStyle:
                GoogleFonts.workSans(fontSize: 13, color: c.textTertiary),
            prefixIcon: Icon(Icons.add_circle_outline_rounded,
                color: c.textTertiary, size: 18),
            contentPadding:
                const EdgeInsets.symmetric(vertical: 12, horizontal: 14),
            isDense: true,
          ),
          onFieldSubmitted: (_) => onAdd(),
          textInputAction: TextInputAction.done,
        ),
      ),
      const SizedBox(width: 8),
      ElevatedButton(
        onPressed: onAdd,
        style: ElevatedButton.styleFrom(
          padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 14),
          shape:
              RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
        ),
        child: Text('Add',
            style: GoogleFonts.spaceGrotesk(
                fontSize: 13, fontWeight: FontWeight.w700)),
      ),
    ]);
  }

  /// Neutral chip with an × remove button (used for skills + certs lists)
  Widget _removableNeutralChip(
    BuildContext context,
    String label, {
    required VoidCallback onRemove,
    Color? color,
  }) {
    final c = AppColors.of(context);
    final chipColor = color ?? c.brand;
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
      decoration: BoxDecoration(
        color: chipColor.withOpacity(0.18),
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: c.cardBorder, width: 1.75),
      ),
      child: Row(mainAxisSize: MainAxisSize.min, children: [
        Text(label,
            style: GoogleFonts.workSans(
                fontSize: 12, color: chipColor, fontWeight: FontWeight.w500)),
        const SizedBox(width: 6),
        GestureDetector(
          onTap: onRemove,
          child: Icon(Icons.close_rounded, size: 13, color: chipColor),
        ),
      ]),
    );
  }

  Widget _metaRow(BuildContext context, String key, String value) {
    final c = AppColors.of(context);
    return Row(mainAxisAlignment: MainAxisAlignment.spaceBetween, children: [
      Text(key, style: Theme.of(context).textTheme.bodyMedium),
      Text(value,
          style: GoogleFonts.workSans(
              fontSize: 12,
              color: c.textSecondary,
              fontWeight: FontWeight.w500)),
    ]);
  }

  Widget _emptyBox(BuildContext context, String msg) {
    final c = AppColors.of(context);
    return Container(
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: c.navy800,
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: c.cardBorder, width: 2.5),
      ),
      child: Text(msg, style: Theme.of(context).textTheme.bodyMedium),
    );
  }
}

// ── Career match expandable card ──────────────────────────────────────────────

class _Top5Card extends StatefulWidget {
  final Top5Match match;
  final bool isTop;
  final List<String> mySkills;
  final void Function(String skill) onAddSkill;
  final void Function(String skill) onRemoveSkill;

  const _Top5Card({
    required this.match,
    required this.isTop,
    required this.mySkills,
    required this.onAddSkill,
    required this.onRemoveSkill,
  });

  @override
  State<_Top5Card> createState() => _Top5CardState();
}

class _Top5CardState extends State<_Top5Card>
    with SingleTickerProviderStateMixin {
  bool _expanded = false;
  late AnimationController _ctrl;
  late Animation<double> _expandAnim;
  late List<String> _unmatchedSkills;
  late List<String> _matchedSkills;
  final _addSkillCtrl = TextEditingController();

  @override
  void initState() {
    super.initState();
    _expanded = widget.isTop;

    // Reconcile against the user's CURRENT skills (not just this job's
    // original snapshot). Without this, reopening a cached analysis always
    // showed previously-promoted skills back under "Skills to develop"
    // (with a checkmark overlay) instead of moving them into "Skills you
    // already have", because the snapshot itself never changes — only
    // `widget.mySkills` does.
    final mySkillsLower = widget.mySkills.map((s) => s.toLowerCase()).toSet();
    _matchedSkills = List<String>.from(widget.match.matchedSkills);
    _unmatchedSkills = [];
    for (final s in widget.match.unmatchedSkills) {
      if (mySkillsLower.contains(s.toLowerCase())) {
        if (!_matchedSkills
            .map((m) => m.toLowerCase())
            .contains(s.toLowerCase())) {
          _matchedSkills.add(s);
        }
      } else {
        _unmatchedSkills.add(s);
      }
    }

    _ctrl = AnimationController(
        vsync: this, duration: const Duration(milliseconds: 250));
    _expandAnim = CurvedAnimation(parent: _ctrl, curve: Curves.easeInOut);
    if (_expanded) _ctrl.value = 1.0;
  }

  /// Recomputed live accuracy: matched / (matched + unmatched) * 100,
  /// same formula the backend uses (matched / ground_truth_count).
  /// Updates instantly whenever a skill is promoted or removed.
  String get _liveAccuracy {
    final total = _matchedSkills.length + _unmatchedSkills.length;
    if (total == 0) return widget.match.nlpAccuracy;
    final pct = _matchedSkills.length / total * 100;
    return '${pct.toStringAsFixed(2)}%';
  }

  @override
  void dispose() {
    _ctrl.dispose();
    _addSkillCtrl.dispose();
    super.dispose();
  }

  void _toggle() {
    setState(() => _expanded = !_expanded);
    _expanded ? _ctrl.forward() : _ctrl.reverse();
  }

  void _addMissingSkill() {
    final skill = _addSkillCtrl.text.trim();
    if (skill.isEmpty) return;
    if (_unmatchedSkills
        .map((s) => s.toLowerCase())
        .contains(skill.toLowerCase())) {
      _addSkillCtrl.clear();
      return;
    }
    setState(() => _unmatchedSkills.add(skill));
    _addSkillCtrl.clear();
  }

  /// Move a skill back down into "Skills to develop" and remove it from
  /// the user's profile — the inverse of _promoteSkill.
  void _demoteSkill(String skill) {
    widget.onRemoveSkill(skill);
    setState(() {
      _matchedSkills.remove(skill);
      if (!_unmatchedSkills
          .map((s) => s.toLowerCase())
          .contains(skill.toLowerCase())) {
        _unmatchedSkills.add(skill);
      }
    });
  }

  /// Move a missing skill up into the user's skills list, and into this
  /// job's "Skills you already have" section, then let the accuracy
  /// ring recompute against the new matched/unmatched split.
  void _promoteSkill(String skill) {
    widget.onAddSkill(skill);
    setState(() {
      _unmatchedSkills.remove(skill);
      if (!_matchedSkills
          .map((s) => s.toLowerCase())
          .contains(skill.toLowerCase())) {
        _matchedSkills.add(skill);
      }
    });
  }

  Color get _rankColor => switch (widget.match.rank) {
        1 => AppTheme.teal500,
        2 => AppTheme.green500,
        3 => AppTheme.amber500,
        _ => AppColors.of(context).textTertiary,
      };

  @override
  Widget build(BuildContext context) {
    final m = widget.match;
    final c = AppColors.of(context);

    return Container(
      decoration: BoxDecoration(
        color: c.navy800,
        borderRadius: BorderRadius.circular(AppTheme.radius),
        border: Border.all(
          color: c.cardBorder,
          width: widget.isTop ? AppTheme.borderWidthThick : AppTheme.borderWidth,
        ),
        boxShadow: AppTheme.hardShadow(c.cardBorder,
            dx: widget.isTop ? 5 : 3, dy: widget.isTop ? 5 : 3),
      ),
      child: Column(children: [
        // ── Header row ────────────────────────────────────────────────
        GestureDetector(
          onTap: _toggle,
          behavior: HitTestBehavior.opaque,
          child: Padding(
            padding: const EdgeInsets.all(14),
            child: Row(children: [
              Container(
                width: 32,
                height: 32,
                decoration: BoxDecoration(
                  color: _rankColor,
                  borderRadius: BorderRadius.circular(8),
                  border: Border.all(color: c.cardBorder, width: 1.5),
                ),
                child: Center(
                  child: Text('#${m.rank}',
                      style: GoogleFonts.spaceGrotesk(
                          fontSize: 12,
                          fontWeight: FontWeight.w700,
                          color: AppTheme.ink)),
                ),
              ),
              const SizedBox(width: 12),
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
                    style: GoogleFonts.workSans(
                        fontSize: 11, color: c.textTertiary),
                  ),
                ],
              )),
              const SizedBox(width: 10),
              AccuracyRing(accuracy: _liveAccuracy, size: 48),
              const SizedBox(width: 8),
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
                        color: c.brand.withOpacity(0.16),
                        borderRadius: BorderRadius.circular(10),
                        border: Border.all(color: c.cardBorder, width: 1.75),
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
                              style: GoogleFonts.workSans(
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
                  if (_matchedSkills.isNotEmpty) ...[
                    _sectionLabel(
                        context,
                        Icons.check_circle_outline_rounded,
                        AppTheme.green500,
                        'Skills you already have (${_matchedSkills.length})'),
                    const SizedBox(height: 8),
                    Wrap(
                        spacing: 6,
                        runSpacing: 6,
                        children: _matchedSkills
                            .map((s) => _matchedSkillChip(context, s))
                            .toList()),
                    const SizedBox(height: 14),
                  ],

                  // ── Skills to develop (tap + to add to your skills) ───
                  _sectionLabel(
                      context,
                      Icons.radio_button_unchecked_rounded,
                      AppTheme.red500,
                      'Skills to develop (${_unmatchedSkills.length})'),
                  const SizedBox(height: 4),
                  Text(
                    'Tap + on any skill to add it to your profile',
                    style: GoogleFonts.workSans(
                        fontSize: 11, color: c.textTertiary),
                  ),
                  const SizedBox(height: 8),

                  // Add missing skill text input
                  Row(children: [
                    Expanded(
                      child: TextFormField(
                        controller: _addSkillCtrl,
                        style: GoogleFonts.workSans(
                            fontSize: 12, color: c.textPrimary),
                        decoration: InputDecoration(
                          hintText: 'Add a missing skill…',
                          hintStyle: GoogleFonts.workSans(
                              fontSize: 12, color: c.textTertiary),
                          prefixIcon: Icon(Icons.add_rounded,
                              color: c.textTertiary, size: 16),
                          contentPadding: const EdgeInsets.symmetric(
                              vertical: 10, horizontal: 12),
                          isDense: true,
                        ),
                        onFieldSubmitted: (_) => _addMissingSkill(),
                        textInputAction: TextInputAction.done,
                      ),
                    ),
                    const SizedBox(width: 6),
                    GestureDetector(
                      onTap: _addMissingSkill,
                      child: Container(
                        padding: const EdgeInsets.symmetric(
                            horizontal: 12, vertical: 10),
                        decoration: BoxDecoration(
                          color: c.brand,
                          borderRadius: BorderRadius.circular(8),
                          border: Border.all(color: c.cardBorder, width: 1.75),
                        ),
                        child: Text('Add',
                            style: GoogleFonts.spaceGrotesk(
                                fontSize: 12,
                                fontWeight: FontWeight.w700,
                                color: Theme.of(context).brightness ==
                                        Brightness.dark
                                    ? AppColors.of(context).navy900
                                    : Colors.white)),
                      ),
                    ),
                  ]),
                  const SizedBox(height: 8),

                  if (_unmatchedSkills.isEmpty)
                    Text('No missing skills — great!',
                        style: GoogleFonts.workSans(
                            fontSize: 12, color: c.textTertiary))
                  else
                    Wrap(
                        spacing: 6,
                        runSpacing: 6,
                        children: _unmatchedSkills
                            .map((s) => _missingSkillChip(context, s))
                            .toList()),
                ],
              ),
            ),
          ]),
        ),
      ]),
    );
  }

  /// Skill you already have — shows × to move it back to "Skills to develop"
  Widget _matchedSkillChip(BuildContext context, String skill) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 9, vertical: 5),
      decoration: BoxDecoration(
        color: AppTheme.teal200,
        borderRadius: BorderRadius.circular(7),
        border: Border.all(color: AppColors.of(context).cardBorder, width: 1.5),
      ),
      child: Row(mainAxisSize: MainAxisSize.min, children: [
        Text(skill,
            style: GoogleFonts.workSans(
                fontSize: 11,
                color: AppTheme.tealDark,
                fontWeight: FontWeight.w600)),
        const SizedBox(width: 5),
        // × button — moves the skill back to "Skills to develop"
        GestureDetector(
          onTap: () => _demoteSkill(skill),
          child: const Icon(Icons.close_rounded,
              size: 13, color: AppTheme.tealDark),
        ),
      ]),
    );
  }

  /// Missing skill chip — shows + to promote it to "Skills you already have"
  Widget _missingSkillChip(BuildContext context, String skill) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 9, vertical: 5),
      decoration: BoxDecoration(
        color: AppTheme.red400.withOpacity(0.28),
        borderRadius: BorderRadius.circular(7),
        border: Border.all(color: AppColors.of(context).cardBorder, width: 1.5),
      ),
      child: Row(mainAxisSize: MainAxisSize.min, children: [
        Text(skill,
            style: GoogleFonts.workSans(
                fontSize: 11,
                color: Theme.of(context).brightness == Brightness.dark
                    ? AppTheme.red400
                    : const Color(0xFF8C2A1C),
                fontWeight: FontWeight.w600)),
        const SizedBox(width: 5),
        // + button — promotes the skill to "Skills you already have"
        GestureDetector(
          onTap: () => _promoteSkill(skill),
          child: Icon(Icons.add_circle_rounded,
              size: 14,
              color: Theme.of(context).brightness == Brightness.dark
                  ? AppTheme.red400
                  : const Color(0xFF8C2A1C)),
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
