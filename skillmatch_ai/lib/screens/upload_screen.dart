// lib/screens/upload_screen.dart
// Analysis mode: Targeted | Full  ×  File mode: PDF | Image | Scan

import 'dart:io';
import 'package:file_picker/file_picker.dart';
import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:image_picker/image_picker.dart';

import '../models/models.dart';
import '../services/api_service.dart';
import '../services/firestore_service.dart';
import '../theme.dart';
import '../widgets/common_widgets.dart';
import 'camera_scan_screen.dart';
import 'full_results_screen.dart';
import 'results_screen.dart';

enum FileMode { pdf, image, scan }

class UploadScreen extends StatefulWidget {
  final ResumeUser user;
  const UploadScreen({super.key, required this.user});

  @override
  State<UploadScreen> createState() => _UploadScreenState();
}

class _UploadScreenState extends State<UploadScreen> with TickerProviderStateMixin {
  final _targetJobCtrl = TextEditingController();
  final _formKey       = GlobalKey<FormState>();

  // Two top-level modes
  AnalysisType _analysisType = AnalysisType.targeted;
  FileMode     _fileMode     = FileMode.pdf;

  // Per file-mode payloads
  PlatformFile? _pickedPdf;
  XFile?        _pickedImage;
  String?       _scannedText;
  int           _scannedPageCount = 0;
  List<File>    _scannedFiles     = [];

  bool   _isAnalyzing   = false;
  String _statusMessage = '';
  String? _error;

  final _api           = ApiService();
  final _firestore     = FirestoreService();
  final _imagePicker   = ImagePicker();

  late AnimationController _pulseCtrl;
  late Animation<double>   _pulseAnim;

  @override
  void initState() {
    super.initState();
    _pulseCtrl = AnimationController(
        vsync: this, duration: const Duration(milliseconds: 1800))
      ..repeat(reverse: true);
    _pulseAnim = Tween<double>(begin: 0.97, end: 1.0)
        .animate(CurvedAnimation(parent: _pulseCtrl, curve: Curves.easeInOut));
  }

  @override
  void dispose() {
    _pulseCtrl.dispose();
    _targetJobCtrl.dispose();
    super.dispose();
  }

  // ── File picking ──────────────────────────────────────────────────────────

  Future<void> _pickPdf() async {
    final r = await FilePicker.platform.pickFiles(
      type: FileType.custom,
      allowedExtensions: ['pdf'],
      withData: true,        // load bytes immediately — no async path copy
      allowCompression: false,
    );
    if (r != null && r.files.isNotEmpty) {
      setState(() { _pickedPdf = r.files.first; _error = null; });
    }
  }

  Future<void> _pickImage() async {
    final x = await _imagePicker.pickImage(
        source: ImageSource.gallery, imageQuality: 95);
    if (x != null) setState(() { _pickedImage = x; _error = null; });
  }

  Future<void> _openScanner() async {
    final result = await Navigator.push<Map<String, dynamic>>(
        context, MaterialPageRoute(builder: (_) => const CameraScanScreen()));
    if (result != null && mounted) {
      setState(() {
        _scannedText      = result['text']       as String?;
        _scannedPageCount = result['page_count'] as int? ?? 0;
        _scannedFiles     = List<File>.from(result['files'] ?? []);
        _error            = null;
      });
    }
  }

  // ── Helpers ───────────────────────────────────────────────────────────────

  bool get _hasPayload => switch (_fileMode) {
    FileMode.pdf   => _pickedPdf != null,
    FileMode.image => _pickedImage != null,
    FileMode.scan  => (_scannedText ?? '').isNotEmpty,
  };

  String get _fileName => switch (_fileMode) {
    FileMode.pdf   => _pickedPdf?.name  ?? 'resume.pdf',
    FileMode.image => _pickedImage?.name ?? 'resume.jpg',
    FileMode.scan  => 'scan_${DateTime.now().millisecondsSinceEpoch}.txt',
  };

  String get _missingMsg => switch (_fileMode) {
    FileMode.pdf   => 'Please pick a PDF resume first.',
    FileMode.image => 'Please pick an image of your resume.',
    FileMode.scan  => 'Please scan at least one page first.',
  };

  // ── Analysis ──────────────────────────────────────────────────────────────

  Future<void> _analyze() async {
    if (_analysisType == AnalysisType.targeted) {
      if (!_formKey.currentState!.validate()) return;
    }
    if (!_hasPayload) { setState(() => _error = _missingMsg); return; }

    setState(() { _isAnalyzing = true; _error = null; _statusMessage = 'Checking cache…'; });

    try {
      if (_analysisType == AnalysisType.targeted) {
        await _runTargeted();
      } else {
        await _runFull();
      }
    } catch (e) {
      setState(() => _error = e.toString().replaceFirst('Exception: ', ''));
    } finally {
      if (mounted) setState(() => _isAnalyzing = false);
    }
  }

  // ── Targeted flow ─────────────────────────────────────────────────────────

  Future<void> _runTargeted() async {
    final fileName  = _fileName;
    final targetJob = _targetJobCtrl.text.trim();
    final jobId     = targetJob.toLowerCase().replaceAll(' ', '_');

    // Cache check
    final cached = await _firestore.findCachedScan(
        userId: widget.user.id, fileName: fileName, targetJobId: jobId);
    if (cached != null && mounted) {
      setState(() => _statusMessage = 'Loaded from cache!');
      await Future.delayed(const Duration(milliseconds: 350));
      if (mounted) _navTargeted(cached, fromCache: true);
      return;
    }

    final result = await _callApi<TargetCareerResult>(() async {
      switch (_fileMode) {
        case FileMode.pdf:
          setState(() => _statusMessage = 'Extracting & analyzing (~30s)…');
          return _api.analyzeTargetCareer(
              filePath:  _pickedPdf!.path ?? '',
              fileName:  fileName,
              targetJob: targetJob,
              fileBytes: _pickedPdf!.bytes);
        case FileMode.image:
        // Backend runs EasyOCR on the image — send it directly.
          setState(() => _statusMessage = 'Uploading image for analysis…');
          return _api.analyzeTargetCareerImage(
              imagePath: _pickedImage!.path, fileName: fileName, targetJob: targetJob);
        case FileMode.scan:
        // ML Kit already extracted text on-device in camera_scan_screen.
        // Send that text as a plain-text payload.
          setState(() => _statusMessage = 'Sending scanned text…');
          return _api.analyzeTargetCareerScanned(
              scannedText: _scannedText!, fileName: fileName, targetJob: targetJob);
      }
    });

    setState(() => _statusMessage = 'Saving to Firestore…');
    final record = ResumeScanRecord(
      id: '', userId: widget.user.id, fileName: fileName,
      analysisType:   AnalysisType.targeted,
      uploadMode:     _fileMode.name,
      targetJobTitle: result.targetJob.title,
      targetJobId:    result.targetJob.jobId,
      industry:       result.targetJob.industry,
      skills:         result.resumeSkills.discoveredSkills,
      certs:          result.resumeSkills.discoveredCerts,
      matchedSkills:  result.skillGap.matchedSkills,
      missingSkills:  result.skillGap.missingSkills,
      nlpAccuracy:    result.skillGap.nlpAccuracy,
      hybridScore:    result.skillGap.hybridScore,
      scannedAt:      DateTime.now(),
    );
    final docId = await _firestore.saveScanRecord(record);
    if (mounted) _navTargeted(record.copyWith(id: docId), fromCache: false);
  }

  // ── Full analysis flow ────────────────────────────────────────────────────

  Future<void> _runFull() async {
    final fileName = _fileName;

    // Full analysis cache check (no jobId — use special sentinel)
    final cached = await _firestore.findCachedScan(
        userId: widget.user.id, fileName: fileName, targetJobId: '__full__');
    if (cached != null && mounted) {
      setState(() => _statusMessage = 'Loaded from cache!');
      await Future.delayed(const Duration(milliseconds: 350));
      if (mounted) _navFull(cached, fromCache: true);
      return;
    }

    final result = await _callApi<FullAnalysisResult>(() async {
      switch (_fileMode) {
        case FileMode.pdf:
          setState(() => _statusMessage = 'Full pipeline analysis (~45s)…');
          return _api.analyzeFullResume(
              filePath:  _pickedPdf!.path ?? '',
              fileName:  fileName,
              fileBytes: _pickedPdf!.bytes);
        case FileMode.image:
        // Backend runs EasyOCR on the image — send it directly.
          setState(() => _statusMessage = 'Uploading image for full analysis…');
          return _api.analyzeFullResumeImage(imagePath: _pickedImage!.path, fileName: fileName);
        case FileMode.scan:
        // ML Kit already extracted text on-device — send as plain text.
          setState(() => _statusMessage = 'Sending scanned text for full analysis…');
          return _api.analyzeFullResumeScanned(scannedText: _scannedText!, fileName: fileName);
      }
    });

    setState(() => _statusMessage = 'Saving to Firestore…');

    // Best match is top5[0]
    final best = result.top5.isNotEmpty ? result.top5.first : null;

    // Stage 3 matched_skills are the fully processed and deduplicated skills
    // validated by the backend against the corpus — use these, not Stage 2 raw output.
    final processedSkills = result.matchedSkills;

    final record = ResumeScanRecord(
      id: '', userId: widget.user.id, fileName: fileName,
      analysisType:   AnalysisType.full,
      uploadMode:     _fileMode.name,
      skills:         processedSkills,
      certs:          result.discoveredCerts,
      top5Matches:    result.top5,
      faissScore:     result.faissScore,
      nlpAccuracy:    best?.nlpAccuracy ?? '0%',
      matchedSkills:  best?.matchedSkills ?? [],
      missingSkills:  best?.unmatchedSkills ?? [],
      targetJobTitle: best?.title ?? '',
      targetJobId:    '__full__',
      scannedAt:      DateTime.now(),
    );
    final docId = await _firestore.saveScanRecord(record);
    if (mounted) _navFull(record.copyWith(id: docId), fromCache: false);
  }

  Future<T> _callApi<T>(Future<T> Function() fn) async => fn();

  void _navTargeted(ResumeScanRecord r, {required bool fromCache}) =>
      Navigator.push(context, MaterialPageRoute(
          builder: (_) => ResultsScreen(record: r, user: widget.user, fromCache: fromCache)));

  void _navFull(ResumeScanRecord r, {required bool fromCache}) =>
      Navigator.push(context, MaterialPageRoute(
          builder: (_) => FullResultsScreen(record: r, user: widget.user, fromCache: fromCache)));

  // ── Build ─────────────────────────────────────────────────────────────────

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: SafeArea(
        child: SingleChildScrollView(
          padding: const EdgeInsets.symmetric(horizontal: 22, vertical: 18),
          child: Form(
            key: _formKey,
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                _header(),
                const SizedBox(height: 24),

                // ── Analysis type toggle (primary) ─────────────────────────
                _analysisTypeToggle(),
                const SizedBox(height: 8),
                _analysisTypeDescription(),
                const SizedBox(height: 20),

                // ── File mode tabs (secondary) ─────────────────────────────
                _fileModeSelector(),
                const SizedBox(height: 16),

                // ── Drop zone ──────────────────────────────────────────────
                AnimatedSwitcher(
                  duration: const Duration(milliseconds: 220),
                  transitionBuilder: (child, anim) => FadeTransition(
                    opacity: anim,
                    child: SlideTransition(
                        position: Tween(begin: const Offset(0, 0.04), end: Offset.zero)
                            .animate(anim),
                        child: child),
                  ),
                  child: _dropZone(),
                ),
                const SizedBox(height: 20),

                // ── Target job field — only in targeted mode ───────────────
                AnimatedSize(
                  duration: const Duration(milliseconds: 250),
                  curve: Curves.easeInOut,
                  child: _analysisType == AnalysisType.targeted
                      ? _targetJobField()
                      : const SizedBox.shrink(),
                ),
                const SizedBox(height: 24),

                if (_error != null) _errorBox(),
                _isAnalyzing ? _analyzingState() : _analyzeBtn(),
                const SizedBox(height: 32),
              ],
            ),
          ),
        ),
      ),
    );
  }

  // ── Header ────────────────────────────────────────────────────────────────

  Widget _header() => Row(children: [
    AvatarCircle(initials: widget.user.avatarInitials, size: 40),
    const SizedBox(width: 12),
    Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
      Text('Hi, ${widget.user.name.split(' ').first} 👋',
          style: Theme.of(context).textTheme.headlineMedium),
      Text('Upload & analyze your resume',
          style: Theme.of(context).textTheme.bodyMedium),
    ]),
  ]);

  // ── Analysis type toggle ─────────────────────────────────────────────────

  Widget _analysisTypeToggle() {
    return Row(children: [
      Expanded(child: _analysisTab(AnalysisType.targeted,
          Icons.gps_fixed_rounded, 'Target Career')),
      const SizedBox(width: 10),
      Expanded(child: _analysisTab(AnalysisType.full,
          Icons.grid_view_rounded, 'Full Analysis')),
    ]);
  }

  Widget _analysisTab(AnalysisType type, IconData icon, String label) {
    final sel = _analysisType == type;
    return GestureDetector(
      onTap: () => setState(() { _analysisType = type; _error = null; }),
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 220),
        padding: const EdgeInsets.symmetric(vertical: 14),
        decoration: BoxDecoration(
          color: sel ? AppTheme.teal500 : AppTheme.navy800,
          borderRadius: BorderRadius.circular(14),
          border: Border.all(
            color: sel ? AppTheme.teal500 : AppTheme.cardBorder,
            width: sel ? 0 : 1,
          ),
          boxShadow: sel
              ? [BoxShadow(color: AppTheme.teal500.withOpacity(0.25),
              blurRadius: 16, offset: const Offset(0, 4))]
              : [],
        ),
        child: Column(mainAxisSize: MainAxisSize.min, children: [
          Icon(icon, size: 20, color: sel ? AppTheme.navy900 : AppTheme.textTertiary),
          const SizedBox(height: 5),
          Text(label, style: GoogleFonts.spaceGrotesk(
              fontSize: 13, fontWeight: FontWeight.w700,
              color: sel ? AppTheme.navy900 : AppTheme.textSecondary)),
        ]),
      ),
    );
  }

  Widget _analysisTypeDescription() {
    final isTargeted = _analysisType == AnalysisType.targeted;
    return AnimatedSwitcher(
      duration: const Duration(milliseconds: 200),
      child: Container(
        key: ValueKey(_analysisType),
        padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
        decoration: BoxDecoration(
          color: AppTheme.navy800,
          borderRadius: BorderRadius.circular(10),
          border: Border.all(color: AppTheme.cardBorder),
        ),
        child: Row(children: [
          Icon(
            isTargeted ? Icons.gps_fixed_rounded : Icons.grid_view_rounded,
            size: 14, color: AppTheme.teal500,
          ),
          const SizedBox(width: 8),
          Expanded(child: Text(
            isTargeted
                ? 'Match your resume against one specific job title. See exactly which skills you have and which are missing.'
                : 'Run the full pipeline — extracts all skills, runs NLP validation, returns Top 5 best-matched careers with FAISS integrity check.',
            style: GoogleFonts.inter(fontSize: 12, color: AppTheme.textSecondary, height: 1.5),
          )),
        ]),
      ),
    );
  }

  // ── File mode selector ────────────────────────────────────────────────────

  Widget _fileModeSelector() => Container(
    decoration: BoxDecoration(
      color: AppTheme.navy800,
      borderRadius: BorderRadius.circular(12),
      border: Border.all(color: AppTheme.cardBorder),
    ),
    padding: const EdgeInsets.all(3),
    child: Row(children: [
      _fileTab(FileMode.pdf,   Icons.picture_as_pdf_outlined,  'PDF'),
      _fileTab(FileMode.image, Icons.image_outlined,            'Image'),
      _fileTab(FileMode.scan,  Icons.document_scanner_outlined, 'Scan'),
    ]),
  );

  Widget _fileTab(FileMode m, IconData icon, String label) {
    final sel = _fileMode == m;
    return Expanded(
      child: GestureDetector(
        onTap: () => setState(() { _fileMode = m; _error = null; }),
        child: AnimatedContainer(
          duration: const Duration(milliseconds: 180),
          padding: const EdgeInsets.symmetric(vertical: 9),
          decoration: BoxDecoration(
            color: sel ? AppTheme.navy600 : Colors.transparent,
            borderRadius: BorderRadius.circular(9),
            border: sel ? Border.all(color: AppTheme.teal500.withOpacity(0.4), width: 0.8) : null,
          ),
          child: Row(mainAxisAlignment: MainAxisAlignment.center, children: [
            Icon(icon, size: 14,
                color: sel ? AppTheme.teal400 : AppTheme.textTertiary),
            const SizedBox(width: 5),
            Text(label, style: GoogleFonts.spaceGrotesk(
                fontSize: 12, fontWeight: FontWeight.w600,
                color: sel ? AppTheme.teal400 : AppTheme.textTertiary)),
          ]),
        ),
      ),
    );
  }

  // ── Drop zones ────────────────────────────────────────────────────────────

  Widget _dropZone() => switch (_fileMode) {
    FileMode.pdf   => _pdfZone(),
    FileMode.image => _imageZone(),
    FileMode.scan  => _scanZone(),
  };

  Widget _pdfZone() {
    final has = _pickedPdf != null;
    return GestureDetector(
        key: const ValueKey('pdf'), onTap: _isAnalyzing ? null : _pickPdf,
        child: AnimatedBuilder(animation: _pulseAnim,
            builder: (_, child) => Transform.scale(scale: has ? 1.0 : _pulseAnim.value, child: child),
            child: _shell(has, has
                ? _previewContent(Icons.picture_as_pdf_rounded, AppTheme.red400,
                _pickedPdf!.name,
                '${(_pickedPdf!.size / 1024).toStringAsFixed(1)} KB  •  tap to change')
                : _prompt(Icons.upload_file_outlined, 'Tap to upload PDF', 'PDF files only'))));
  }

  Widget _imageZone() {
    final has = _pickedImage != null;
    return GestureDetector(
        key: const ValueKey('image'), onTap: _isAnalyzing ? null : _pickImage,
        child: AnimatedBuilder(animation: _pulseAnim,
            builder: (_, child) => Transform.scale(scale: has ? 1.0 : _pulseAnim.value, child: child),
            child: has ? _imagePrev() : _shell(false,
                _prompt(Icons.image_outlined, 'Tap to pick image', 'JPG, PNG — backend runs OCR'))));
  }

  Widget _imagePrev() => Stack(children: [
    ClipRRect(borderRadius: BorderRadius.circular(16),
        child: Image.file(File(_pickedImage!.path),
            width: double.infinity, height: 190, fit: BoxFit.cover)),
    Positioned.fill(child: Container(
      decoration: BoxDecoration(
        borderRadius: BorderRadius.circular(16),
        gradient: LinearGradient(begin: Alignment.topCenter, end: Alignment.bottomCenter,
            colors: [Colors.transparent, AppTheme.navy900.withOpacity(0.78)]),
      ),
    )),
    Positioned(bottom: 12, left: 14, right: 14, child: Row(children: [
      Expanded(child: Text(_pickedImage!.name,
          style: GoogleFonts.inter(fontSize: 12, color: Colors.white,
              fontWeight: FontWeight.w500), overflow: TextOverflow.ellipsis)),
      GestureDetector(onTap: _pickImage,
          child: Container(
              padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
              decoration: BoxDecoration(color: AppTheme.teal500,
                  borderRadius: BorderRadius.circular(6)),
              child: Text('Change', style: GoogleFonts.inter(
                  fontSize: 11, color: AppTheme.navy900, fontWeight: FontWeight.w600)))),
    ])),
  ]);

  Widget _scanZone() {
    final has = _scannedPageCount > 0;
    return GestureDetector(
        key: const ValueKey('scan'), onTap: _isAnalyzing ? null : _openScanner,
        child: _shell(has, has ? _scanPreview() :
        _prompt(Icons.document_scanner_outlined, 'Tap to open camera',
            'Scan pages • on-device ML Kit OCR')));
  }

  Widget _scanPreview() => Column(mainAxisAlignment: MainAxisAlignment.center, children: [
    if (_scannedFiles.isNotEmpty)
      SizedBox(height: 72, child: ListView.separated(
        scrollDirection: Axis.horizontal, shrinkWrap: true,
        itemCount: _scannedFiles.length.clamp(0, 5),
        separatorBuilder: (_, __) => const SizedBox(width: 6),
        itemBuilder: (_, i) => ClipRRect(borderRadius: BorderRadius.circular(6),
            child: Image.file(_scannedFiles[i], width: 50, height: 72, fit: BoxFit.cover)),
      )),
    const SizedBox(height: 10),
    Row(mainAxisAlignment: MainAxisAlignment.center, children: [
      const Icon(Icons.check_circle_rounded, color: AppTheme.teal500, size: 16),
      const SizedBox(width: 6),
      Text('$_scannedPageCount page${_scannedPageCount != 1 ? 's' : ''} scanned',
          style: GoogleFonts.spaceGrotesk(fontSize: 14,
              fontWeight: FontWeight.w600, color: AppTheme.teal400)),
    ]),
    const SizedBox(height: 4),
    Text('Tap to re-scan',
        style: GoogleFonts.inter(fontSize: 12, color: AppTheme.textTertiary)),
  ]);

  Widget _shell(bool has, Widget child) => AnimatedContainer(
    duration: const Duration(milliseconds: 280),
    width: double.infinity,
    constraints: const BoxConstraints(minHeight: 155),
    padding: const EdgeInsets.all(20),
    decoration: BoxDecoration(
      color: has ? AppTheme.teal500.withOpacity(0.05) : AppTheme.navy800,
      borderRadius: BorderRadius.circular(16),
      border: Border.all(
          color: has ? AppTheme.teal500 : AppTheme.navy500,
          width: has ? 1.5 : 1),
    ),
    child: child,
  );

  Widget _prompt(IconData icon, String title, String sub) =>
      Column(mainAxisAlignment: MainAxisAlignment.center, children: [
        Container(width: 52, height: 52,
            decoration: BoxDecoration(color: AppTheme.navy700,
                borderRadius: BorderRadius.circular(14),
                border: Border.all(color: AppTheme.cardBorder)),
            child: Icon(icon, color: AppTheme.textTertiary, size: 24)),
        const SizedBox(height: 12),
        Text(title, style: GoogleFonts.spaceGrotesk(
            fontSize: 15, fontWeight: FontWeight.w600, color: AppTheme.textSecondary)),
        const SizedBox(height: 4),
        Text(sub, style: GoogleFonts.inter(fontSize: 12, color: AppTheme.textTertiary)),
      ]);

  Widget _previewContent(IconData icon, Color c, String name, String meta) =>
      Column(mainAxisAlignment: MainAxisAlignment.center, children: [
        Container(width: 52, height: 52,
            decoration: BoxDecoration(color: c.withOpacity(0.1),
                borderRadius: BorderRadius.circular(14)),
            child: Icon(icon, color: c, size: 26)),
        const SizedBox(height: 10),
        Text(name, style: GoogleFonts.spaceGrotesk(
            fontSize: 14, fontWeight: FontWeight.w600, color: AppTheme.textPrimary),
            overflow: TextOverflow.ellipsis),
        const SizedBox(height: 4),
        Text(meta, style: GoogleFonts.inter(fontSize: 12, color: AppTheme.textTertiary)),
      ]);

  // ── Target job field ──────────────────────────────────────────────────────

  Widget _targetJobField() => Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
    Text('Target Job Title', style: GoogleFonts.spaceGrotesk(
        fontSize: 13, fontWeight: FontWeight.w600, color: AppTheme.textSecondary)),
    const SizedBox(height: 8),
    TextFormField(
      controller: _targetJobCtrl, enabled: !_isAnalyzing,
      style: GoogleFonts.inter(fontSize: 14, color: AppTheme.textPrimary),
      decoration: const InputDecoration(
        hintText: 'e.g. Data Scientist, Software Engineer',
        prefixIcon: Icon(Icons.work_outline_rounded,
            color: AppTheme.textTertiary, size: 18),
      ),
      validator: (v) => (v == null || v.trim().isEmpty)
          ? 'Enter the target job title' : null,
    ),
  ]);

  // ── Error / loading / button ──────────────────────────────────────────────

  Widget _errorBox() => Container(
    margin: const EdgeInsets.only(bottom: 16),
    padding: const EdgeInsets.all(14),
    decoration: BoxDecoration(
      color: AppTheme.red500.withOpacity(0.08),
      borderRadius: BorderRadius.circular(12),
      border: Border.all(color: AppTheme.red500.withOpacity(0.3)),
    ),
    child: Row(crossAxisAlignment: CrossAxisAlignment.start, children: [
      const Icon(Icons.error_outline_rounded, color: AppTheme.red400, size: 16),
      const SizedBox(width: 10),
      Expanded(child: Text(_error!,
          style: GoogleFonts.inter(fontSize: 13, color: AppTheme.red400))),
    ]),
  );

  Widget _analyzingState() => Column(children: [
    const LinearProgressIndicator(
        backgroundColor: AppTheme.navy700, color: AppTheme.teal500),
    const SizedBox(height: 12),
    Text(_statusMessage,
        style: GoogleFonts.inter(fontSize: 13, color: AppTheme.textSecondary),
        textAlign: TextAlign.center),
  ]);

  Widget _analyzeBtn() {
    final isTargeted = _analysisType == AnalysisType.targeted;
    return SizedBox(
      width: double.infinity,
      child: ElevatedButton.icon(
        onPressed: _analyze,
        icon: Icon(
            isTargeted ? Icons.gps_fixed_rounded : Icons.grid_view_rounded,
            size: 18),
        label: Text(isTargeted ? 'Analyze — Target Career' : 'Analyze — Full Pipeline'),
      ),
    );
  }
}