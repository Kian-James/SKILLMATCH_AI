// lib/screens/upload_screen.dart
// Analysis mode: Targeted | Full  ×  File mode: PDF | Image

import 'dart:io';
import 'dart:typed_data';

import 'package:file_picker/file_picker.dart';
import 'package:flutter/foundation.dart' show kIsWeb;
import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:image_picker/image_picker.dart';
import 'package:syncfusion_flutter_pdf/pdf.dart';
import 'package:wakelock_plus/wakelock_plus.dart';

import '../models/models.dart';
import '../services/api_service.dart';
import '../services/firestore_service.dart';
import '../services/tutorial_service.dart';
import '../theme.dart';
import '../utils/responsive.dart';
import '../widgets/common_widgets.dart';
import '../widgets/tutorial_overlay.dart';
import 'full_results_screen.dart';
import 'results_screen.dart';

enum FileMode { pdf, image }

class UploadScreen extends StatefulWidget {
  final ResumeUser user;
  const UploadScreen({super.key, required this.user});

  @override
  State<UploadScreen> createState() => _UploadScreenState();
}

class _UploadScreenState extends State<UploadScreen>
    with TickerProviderStateMixin {
  // ── Tutorial global keys (one per UI element to spotlight) ────────────────
  final _keyHeader = GlobalKey();
  final _keyAnalysisToggle = GlobalKey();
  final _keyAnalysisDesc = GlobalKey();
  final _keyDropZone = GlobalKey();
  final _keyTargetField = GlobalKey();
  final _keyAnalyzeBtn = GlobalKey();
  final _keyPdfTab = GlobalKey();
  final _keyImageTab = GlobalKey();
  final _keyInfoBtn = GlobalKey();

  bool _showTutorial = false;

  // ── Form state ────────────────────────────────────────────────────────────
  final _targetJobCtrl = TextEditingController();
  final _targetJobFocus = FocusNode();
  final _formKey = GlobalKey<FormState>();

  AnalysisType _analysisType = AnalysisType.targeted;
  FileMode _fileMode = FileMode.pdf;

  PlatformFile? _pickedPdf;
  XFile? _pickedImage;
  Uint8List? _pickedImageBytes;

  bool _isAnalyzing = false;
  String _statusMessage = '';
  String? _error;

  final _api = ApiService();
  final _firestore = FirestoreService();
  final _imagePicker = ImagePicker();

  late AnimationController _pulseCtrl;
  late Animation<double> _pulseAnim;

  // ── Target-career autosuggest ─────────────────────────────────────────────
  // Job titles pulled once from Firestore's resume_scans history, then
  // filtered client-side as the user types. Kept static so it survives
  // across visits to this screen within the same app session instead of
  // re-querying every time.
  static List<String>? _cachedJobTitles;
  List<String> _jobTitleSuggestions = [];
  double _targetFieldWidth = 300;

  @override
  void initState() {
    super.initState();

    _pulseCtrl = AnimationController(
        vsync: this, duration: const Duration(milliseconds: 1800))
      ..repeat(reverse: true);
    _pulseAnim = Tween<double>(begin: 0.97, end: 1.0)
        .animate(CurvedAnimation(parent: _pulseCtrl, curve: Curves.easeInOut));

    // Check first launch after the first frame (keys must be laid out)
    WidgetsBinding.instance.addPostFrameCallback((_) => _checkTutorial());

    _loadJobTitleSuggestions();
  }

  Future<void> _loadJobTitleSuggestions() async {
    if (_cachedJobTitles != null) {
      setState(() => _jobTitleSuggestions = _cachedJobTitles!);
      return;
    }

    final merged = <String>{};

    try {
      final dataset = await _firestore.fetchJobsTitles();
      debugPrint('[autosuggest] jobs collection: ${dataset.length} titles');
      merged.addAll(dataset);
    } catch (e) {
      debugPrint('[autosuggest] jobs collection load failed: $e');
    }

    try {
      final own =
          await _firestore.fetchOwnTargetJobTitles(userId: widget.user.id);
      debugPrint('[autosuggest] own scan history: ${own.length} titles');
      merged.addAll(own);
    } catch (e) {
      debugPrint('[autosuggest] own scan history load failed: $e');
    }

    final titles = merged.toList()
      ..sort((a, b) => a.toLowerCase().compareTo(b.toLowerCase()));
    debugPrint('[autosuggest] total merged: ${titles.length} titles');

    _cachedJobTitles = titles;
    if (mounted) setState(() => _jobTitleSuggestions = titles);
  }

  @override
  void dispose() {
    _pulseCtrl.dispose();
    _targetJobCtrl.dispose();
    _targetJobFocus.dispose();
    if (_isAnalyzing) WakelockPlus.disable();
    super.dispose();
  }

  // ── Tutorial logic ────────────────────────────────────────────────────────

  Future<void> _checkTutorial() async {
    final first = await TutorialService.isFirstLaunch();
    if (first && mounted) {
      setState(() => _showTutorial = true);
    }
  }

  void _finishTutorial() async {
    await TutorialService.markDone();
    if (mounted) setState(() => _showTutorial = false);
  }

  /// Asks whether to overwrite an existing analysis for this exact
  /// (file name, analysis type) pair, or view the existing one instead.
  /// Returns true = override (re-run + overwrite), false/null = view cached.
  Future<bool> _confirmOverride(String fileName) async {
    final c = AppColors.of(context);
    final confirmed = await showDialog<bool>(
          context: context,
          builder: (_) => AlertDialog(
            backgroundColor: c.navy800,
            shape:
                RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
            title: Text('Already analyzed',
                style: GoogleFonts.spaceGrotesk(
                    fontWeight: FontWeight.w700, color: c.textPrimary)),
            content: Text(
              '"$fileName" has already been analyzed. '
              'Are you sure you want to override the existing result? '
              'If not, we\'ll take you to the existing analysis instead.',
              style: GoogleFonts.workSans(
                  fontSize: 13, color: c.textSecondary, height: 1.5),
            ),
            actions: [
              TextButton(
                onPressed: () => Navigator.pop(context, false),
                child: Text('View existing',
                    style: GoogleFonts.workSans(color: c.textSecondary)),
              ),
              ElevatedButton(
                onPressed: () => Navigator.pop(context, true),
                child: const Text('Override'),
              ),
            ],
          ),
        ) ??
        false;
    return confirmed;
  }

  Future<void> _retryTutorial() async {
    final c = AppColors.of(context);
    final confirmed = await showDialog<bool>(
          context: context,
          builder: (_) => AlertDialog(
            backgroundColor: c.navy800,
            shape:
                RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
            title: Text('Show tutorial again?',
                style: GoogleFonts.spaceGrotesk(
                    fontWeight: FontWeight.w700, color: c.textPrimary)),
            content: Text(
              'This will walk you through each feature of the app from the beginning.',
              style: GoogleFonts.workSans(
                  fontSize: 13, color: c.textSecondary, height: 1.5),
            ),
            actions: [
              TextButton(
                onPressed: () => Navigator.pop(context, false),
                child: Text('Cancel',
                    style: GoogleFonts.workSans(color: c.textSecondary)),
              ),
              ElevatedButton(
                onPressed: () => Navigator.pop(context, true),
                child: const Text('Show tutorial'),
              ),
            ],
          ),
        ) ??
        false;

    if (confirmed && mounted) {
      await TutorialService.reset();
      setState(() => _showTutorial = true);
    }
  }

  List<TutorialStep> get _tutorialSteps => [
        TutorialStep(
          targetKey: _keyHeader,
          title: 'Welcome to SkillMatch AI 👋',
          body:
              'This is your home screen. Upload your resume here and we\'ll tell you which careers match your skills — and what to learn next.',
        ),
        TutorialStep(
          targetKey: _keyAnalysisToggle,
          title: 'Choose your analysis mode',
          body:
              'Pick "Target Career" to check how well your resume fits one specific job. Pick "Full Analysis" to see your Top 5 best-matched careers all at once.',
        ),
        TutorialStep(
          targetKey: _keyAnalysisDesc,
          title: 'What each mode does',
          body:
              'This box explains the selected mode. It updates when you switch between Target Career and Full Analysis so you always know what to expect.',
        ),
        TutorialStep(
          targetKey: _keyPdfTab,
          title: 'Upload as PDF',
          body:
              'Tap this tab to upload your resume as a PDF file. We\'ll run OCR on it automatically.',
        ),
        TutorialStep(
          targetKey: _keyImageTab,
          title: 'Upload as an image',
          body:
              'Already have a photo of your resume? Tap here and pick it from your gallery. We\'ll read the text automatically.',
        ),
        TutorialStep(
          targetKey: _keyDropZone,
          title: 'Your resume goes here',
          body:
              'After choosing a file type above, tap this area to pick or capture your resume. You\'ll see a preview once it\'s loaded.',
        ),
        TutorialStep(
          targetKey: _keyTargetField,
          title: 'Enter the job you\'re aiming for',
          body:
              'Type the job title you want — for example "Software Engineer" or "Data Analyst". We\'ll compare your resume directly against that role.',
          shape: SpotlightShape.roundedRect,
        ),
        TutorialStep(
          targetKey: _keyAnalyzeBtn,
          title: 'Run the analysis',
          body:
              'Tap this button when your resume is uploaded and your job title is filled in. Results usually take 15–45 seconds.',
          shape: SpotlightShape.roundedRect,
        ),
        TutorialStep(
          targetKey: _keyInfoBtn,
          title: 'You\'re all set! 🎉',
          body:
              'That\'s everything you need to know. You can always tap this ⓘ button to replay this guide whenever you need a refresher.',
          shape: SpotlightShape.circle,
        ),
      ];

  // ── File picking ──────────────────────────────────────────────────────────

  /// Max allowed pages for an uploaded resume PDF.
  static const int _maxPdfPages = 1;

  Future<void> _pickPdf() async {
    final r = await FilePicker.platform.pickFiles(
      type: FileType.custom,
      allowedExtensions: ['pdf'],
      withData: true,
      allowCompression: false,
    );
    if (r == null || r.files.isEmpty) return;

    final file = r.files.first;
    final bytes = file.bytes;

    // Verify page count before accepting the file. `withData: true` above
    // guarantees bytes are populated on every platform (web included).
    if (bytes == null) {
      setState(() => _error = 'Could not read that PDF. Please try again.');
      return;
    }

    int pageCount;
    try {
      final doc = PdfDocument(inputBytes: bytes);
      pageCount = doc.pages.count;
      doc.dispose();
    } catch (_) {
      setState(() => _error = 'That file doesn\'t look like a valid PDF.');
      return;
    }

    if (pageCount > _maxPdfPages) {
      setState(() => _error =
          'Your resume PDF has $pageCount pages. Please upload a resume that is '
              '$_maxPdfPages page or fewer.');
      return;
    }

    setState(() {
      _pickedPdf = file;
      _error = null;
    });
  }

  Future<void> _pickImage() async {
    final x = await _imagePicker.pickImage(
        source: ImageSource.gallery, imageQuality: 95);
    if (x == null) return;

    // XFile.readAsBytes() works on every platform (including web), unlike
    // dart:io File — so we cache the bytes once here and reuse them both
    // for the preview and for the upload request.
    final bytes = await x.readAsBytes();
    setState(() {
      _pickedImage = x;
      _pickedImageBytes = bytes;
      _error = null;
    });
  }

  // ── Helpers ───────────────────────────────────────────────────────────────

  bool get _hasPayload => switch (_fileMode) {
        FileMode.pdf => _pickedPdf != null,
        FileMode.image => _pickedImage != null,
      };

  String get _fileName => switch (_fileMode) {
        FileMode.pdf => _pickedPdf?.name ?? 'resume.pdf',
        FileMode.image => _pickedImage?.name ?? 'resume.jpg',
      };

  String get _missingMsg => switch (_fileMode) {
        FileMode.pdf => 'Please pick a PDF resume first.',
        FileMode.image => 'Please pick an image of your resume.',
      };

  // ── Analysis ──────────────────────────────────────────────────────────────

  Future<void> _analyze() async {
    if (_analysisType == AnalysisType.targeted) {
      if (!_formKey.currentState!.validate()) return;
    }
    if (!_hasPayload) {
      setState(() => _error = _missingMsg);
      return;
    }

    setState(() {
      _isAnalyzing = true;
      _error = null;
      _statusMessage = 'Checking cache…';
    });

    // Keep the screen awake for the whole analysis — this can take ~30s+
    // and the OS auto-locking mid-scan would silently kill the request.
    await WakelockPlus.enable();

    try {
      if (_analysisType == AnalysisType.targeted) {
        await _runTargeted();
      } else {
        await _runFull();
      }
    } catch (e) {
      setState(() => _error = e.toString().replaceFirst('Exception: ', ''));
    } finally {
      await WakelockPlus.disable();
      if (mounted) setState(() => _isAnalyzing = false);
    }
  }

  Future<void> _runTargeted() async {
    final fileName = _fileName;
    final targetJob = _targetJobCtrl.text.trim();
    final jobId = targetJob.toLowerCase().replaceAll(' ', '_');

    final cached = await _firestore.findCachedScan(
        userId: widget.user.id, fileName: fileName, targetJobId: jobId);
    if (cached != null && mounted) {
      final override = await _confirmOverride(fileName);
      if (!override) {
        if (mounted) _navTargeted(cached, fromCache: true);
        return;
      }
      // else: fall through and re-run the analysis fresh below, then
      // overwrite this same record instead of creating a duplicate.
    }

    final result = await _callApi<TargetCareerResult>(() async {
      switch (_fileMode) {
        case FileMode.pdf:
          setState(() => _statusMessage = 'Extracting & analyzing (~30s)…');
          return _api.analyzeTargetCareer(
              filePath: kIsWeb ? '' : (_pickedPdf!.path ?? ''),
              fileName: fileName,
              targetJob: targetJob,
              fileBytes: _pickedPdf!.bytes);
        case FileMode.image:
          setState(() => _statusMessage = 'Uploading image for analysis…');
          return _api.analyzeTargetCareerImage(
              imagePath: kIsWeb ? _pickedImage!.name : _pickedImage!.path,
              fileName: fileName,
              targetJob: targetJob,
              fileBytes: _pickedImageBytes);
      }
    });

    setState(() => _statusMessage = 'Saving…');

    final mergedSkills = {
      ...?cached?.skills,
      ...result.resumeSkills.discoveredSkills,
    }.toList();
    final mergedCerts = {
      ...?cached?.certs,
      ...result.resumeSkills.discoveredCerts,
    }.toList();

    final record = ResumeScanRecord(
      id: cached?.id ?? '',
      userId: widget.user.id,
      fileName: fileName,
      analysisType: AnalysisType.targeted,
      uploadMode: _fileMode.name,
      targetJobTitle: result.targetJob.title,
      targetJobId: result.targetJob.jobId,
      industry: result.targetJob.industry,
      skills: mergedSkills,
      certs: mergedCerts,
      matchedSkills: result.skillGap.matchedSkills,
      missingSkills: result.skillGap.missingSkills,
      nlpAccuracy: result.skillGap.nlpAccuracy,
      hybridScore: result.skillGap.hybridScore,
      fitReason: result.fitReason,
      scannedAt: DateTime.now(),
    );
    final docId = await _firestore.upsertScanRecord(record);
    if (mounted) _navTargeted(record.copyWith(id: docId), fromCache: false);
  }

  Future<void> _runFull() async {
    final fileName = _fileName;

    // Check for an existing record BEFORE calling the (slow, ~45s) backend.
    // If found, ask whether to override it or just view the existing result
    // — full analysis includes STAGE_3_JOB_MATCHING and
    // STAGE_4_FAISS_INTEGRITY, both per-run pipeline checks rather than
    // properties of the resume's content, so we never silently serve a
    // stale cached result (see prior fix) — but we also shouldn't force a
    // full re-run without asking, since that's a real ~45s cost.
    final existing = await _firestore.findCachedScan(
        userId: widget.user.id, fileName: fileName, targetJobId: '__full__');
    if (existing != null && mounted) {
      final override = await _confirmOverride(fileName);
      if (!override) {
        if (mounted) _navFull(existing, fromCache: true);
        return;
      }
      // else: fall through and re-run the analysis fresh below, then
      // overwrite this same record instead of creating a duplicate.
    }

    final result = await _callApi<FullAnalysisResult>(() async {
      switch (_fileMode) {
        case FileMode.pdf:
          setState(() => _statusMessage = 'Full pipeline analysis (~45s)…');
          return _api.analyzeFullResume(
              filePath: kIsWeb ? '' : (_pickedPdf!.path ?? ''),
              fileName: fileName,
              fileBytes: _pickedPdf!.bytes);
        case FileMode.image:
          setState(() => _statusMessage = 'Uploading image for full analysis…');
          return _api.analyzeFullResumeImage(
              imagePath: kIsWeb ? _pickedImage!.name : _pickedImage!.path,
              fileName: fileName,
              fileBytes: _pickedImageBytes);
      }
    });

    setState(() => _statusMessage = 'Saving…');

    final best = result.top5.isNotEmpty ? result.top5.first : null;

    // Merge in skills the user manually promoted last time — the NLP
    // extraction won't "rediscover" those on its own, so without this a
    // re-analysis would silently drop that work.
    final mergedSkills = {
      ...?existing?.skills,
      ...result.discoveredSkills,
    }.toList();
    final mergedCerts = {
      ...?existing?.certs,
      ...result.discoveredCerts,
    }.toList();

    final record = ResumeScanRecord(
      id: existing?.id ?? '',
      userId: widget.user.id,
      fileName: fileName,
      analysisType: AnalysisType.full,
      uploadMode: _fileMode.name,
      skills: mergedSkills,
      certs: mergedCerts,
      top5Matches: result.top5,
      faissScore: result.faissScore,
      nlpAccuracy: best?.nlpAccuracy ?? '0%',
      matchedSkills: best?.matchedSkills ?? [],
      missingSkills: best?.unmatchedSkills ?? [],
      targetJobTitle: best?.title ?? '',
      targetJobId: '__full__',
      scannedAt: DateTime.now(),
    );
    final docId = await _firestore.upsertScanRecord(record);
    if (mounted) _navFull(record.copyWith(id: docId), fromCache: false);
  }

  Future<T> _callApi<T>(Future<T> Function() fn) => fn();

  void _navTargeted(ResumeScanRecord r, {required bool fromCache}) =>
      Navigator.push(
          context,
          MaterialPageRoute(
              builder: (_) => ResultsScreen(
                  record: r, user: widget.user, fromCache: fromCache)));

  void _navFull(ResumeScanRecord r, {required bool fromCache}) =>
      Navigator.push(
          context,
          MaterialPageRoute(
              builder: (_) => FullResultsScreen(
                  record: r, user: widget.user, fromCache: fromCache)));

  // ── Build ─────────────────────────────────────────────────────────────────

  @override
  Widget build(BuildContext context) {
    final c = AppColors.of(context);

    return Stack(
      children: [
        // ── Main screen ────────────────────────────────────────────────
        Scaffold(
          appBar: AppBar(
            backgroundColor: c.navy900,
            automaticallyImplyLeading: false,
            title: Text('Analyze Resume',
                style: Theme.of(context).textTheme.titleLarge),
            actions: [
              // ── Info / retake tutorial button ──────────────────────
              IconButton(
                onPressed: _retryTutorial,
                tooltip: 'How to use this screen',
                icon: KeyedSubtree(
                  key: _keyInfoBtn,
                  child: Icon(
                    Icons.info_outline_rounded,
                    color: c.textTertiary,
                    size: 20,
                  ),
                ),
              ),
              const SizedBox(width: 4),
            ],
          ),
          body: SafeArea(
            child: ResponsiveBody(
            maxWidth: 720,
            child: SingleChildScrollView(
              padding: EdgeInsets.symmetric(
                  horizontal: responsiveGutter(context, 22), vertical: 18),
              child: Form(
                key: _formKey,
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    _header(),
                    const SizedBox(height: 24),

                    // ── Analysis type toggle ────────────────────────────
                    KeyedSubtree(
                      key: _keyAnalysisToggle,
                      child: _analysisTypeToggle(),
                    ),
                    const SizedBox(height: 8),

                    // ── Description ─────────────────────────────────────
                    KeyedSubtree(
                      key: _keyAnalysisDesc,
                      child: _analysisTypeDescription(),
                    ),
                    const SizedBox(height: 20),

                    // ── File mode tabs ──────────────────────────────────
                    _fileModeSelector(),
                    const SizedBox(height: 16),

                    // ── Drop zone ───────────────────────────────────────
                    KeyedSubtree(
                      key: _keyDropZone,
                      child: AnimatedSwitcher(
                        duration: const Duration(milliseconds: 220),
                        transitionBuilder: (child, anim) => FadeTransition(
                          opacity: anim,
                          child: SlideTransition(
                              position: Tween(
                                      begin: const Offset(0, 0.04),
                                      end: Offset.zero)
                                  .animate(anim),
                              child: child),
                        ),
                        child: _dropZone(),
                      ),
                    ),
                    const SizedBox(height: 20),

                    // ── Target job field ────────────────────────────────
                    AnimatedSize(
                      duration: const Duration(milliseconds: 250),
                      curve: Curves.easeInOut,
                      child: _analysisType == AnalysisType.targeted
                          ? KeyedSubtree(
                              key: _keyTargetField, child: _targetJobField())
                          : const SizedBox.shrink(),
                    ),
                    const SizedBox(height: 24),

                    if (_error != null) _errorBox(),

                    // ── Analyze button ──────────────────────────────────
                    KeyedSubtree(
                      key: _keyAnalyzeBtn,
                      child: _isAnalyzing ? _analyzingState() : _analyzeBtn(),
                    ),
                    const SizedBox(height: 32),
                  ],
                ),
              ),
            ),
            ),
          ),
        ),

        // ── Tutorial overlay — sits on top of everything ───────────────
        if (_showTutorial)
          Positioned.fill(
            child: TutorialOverlay(
              steps: _tutorialSteps,
              onFinish: _finishTutorial,
            ),
          ),
      ],
    );
  }

  // ── Header ────────────────────────────────────────────────────────────────

  Widget _header() => KeyedSubtree(
        key: _keyHeader,
        child: Row(children: [
          AvatarCircle(initials: widget.user.avatarInitials, size: 40),
          const SizedBox(width: 12),
          Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
            Text('Hi, ${widget.user.name.split(' ').first} 👋',
                style: Theme.of(context).textTheme.headlineMedium),
            Text('Upload & analyze your resume',
                style: Theme.of(context).textTheme.bodyMedium),
          ]),
        ]),
      );

  // ── Analysis type toggle ──────────────────────────────────────────────────

  Widget _analysisTypeToggle() {
    return Row(children: [
      Expanded(
          child: _analysisTab(
              AnalysisType.targeted, Icons.gps_fixed_rounded, 'Target Career')),
      const SizedBox(width: 10),
      Expanded(
          child: _analysisTab(
              AnalysisType.full, Icons.grid_view_rounded, 'Full Analysis')),
    ]);
  }

  Widget _analysisTab(AnalysisType type, IconData icon, String label) {
    final c = AppColors.of(context);
    final sel = _analysisType == type;
    return GestureDetector(
      onTap: () => setState(() {
        _analysisType = type;
        _error = null;
      }),
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 220),
        padding: const EdgeInsets.only(bottom: 10),
        decoration: BoxDecoration(
          border: Border(
            bottom: BorderSide(
              color: sel ? c.brand : c.cardBorder,
              width: sel ? 2 : 1,
            ),
          ),
        ),
        child: Row(mainAxisAlignment: MainAxisAlignment.center, children: [
          Icon(icon, size: 16, color: sel ? c.brand : c.textTertiary),
          const SizedBox(width: 6),
          Text(label,
              style: GoogleFonts.spaceGrotesk(
                  fontSize: 13,
                  fontWeight: FontWeight.w600,
                  color: sel ? c.brand : c.textSecondary)),
        ]),
      ),
    );
  }

  Widget _analysisTypeDescription() {
    final c = AppColors.of(context);
    final isTargeted = _analysisType == AnalysisType.targeted;
    return AnimatedSwitcher(
      duration: const Duration(milliseconds: 200),
      child: Container(
        key: ValueKey(_analysisType),
        padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
        decoration: BoxDecoration(
          color: c.navy800,
          borderRadius: BorderRadius.circular(10),
          border: Border.all(color: c.cardBorder, width: 2.5),
        ),
        child: Row(children: [
          Icon(
            isTargeted ? Icons.gps_fixed_rounded : Icons.grid_view_rounded,
            size: 14,
            color: c.brand,
          ),
          const SizedBox(width: 8),
          Expanded(
              child: Text(
            isTargeted
                ? 'Match your resume against one specific job title. See exactly which skills you have and which are missing.'
                : 'Discover your top 5 best-matched careers based on all the skills found in your resume.',
            style: GoogleFonts.workSans(
                fontSize: 12, color: c.textSecondary, height: 1.5),
          )),
        ]),
      ),
    );
  }

  // ── File mode selector ────────────────────────────────────────────────────

  Widget _fileModeSelector() {
    final c = AppColors.of(context);
    return Container(
      decoration: BoxDecoration(
        color: c.navy800,
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: c.cardBorder, width: 2.5),
        boxShadow: AppTheme.hardShadow(c.cardBorder, dx: 4, dy: 4),
      ),
      padding: const EdgeInsets.all(3),
      child: Row(children: [
        _fileTab(
            FileMode.pdf, Icons.picture_as_pdf_outlined, 'PDF', _keyPdfTab),
        _fileTab(FileMode.image, Icons.image_outlined, 'Image', _keyImageTab),
      ]),
    );
  }

  Widget _fileTab(FileMode m, IconData icon, String label, GlobalKey key) {
    final c = AppColors.of(context);
    final sel = _fileMode == m;
    return Expanded(
      child: KeyedSubtree(
        key: key,
        child: GestureDetector(
          onTap: () => setState(() {
            _fileMode = m;
            _error = null;
          }),
          child: AnimatedContainer(
            duration: const Duration(milliseconds: 180),
            padding: const EdgeInsets.symmetric(vertical: 9),
            decoration: BoxDecoration(
              color: sel ? c.navy600 : Colors.transparent,
              borderRadius: BorderRadius.circular(9),
              border: sel
                  ? Border.all(color: c.brand.withOpacity(0.4), width: 0.8)
                  : null,
            ),
            child: Row(mainAxisAlignment: MainAxisAlignment.center, children: [
              Icon(icon, size: 14, color: sel ? c.brandSoft : c.textTertiary),
              const SizedBox(width: 5),
              Text(label,
                  style: GoogleFonts.spaceGrotesk(
                      fontSize: 12,
                      fontWeight: FontWeight.w600,
                      color: sel ? c.brandSoft : c.textTertiary)),
            ]),
          ),
        ),
      ),
    );
  }

  // ── Drop zones ────────────────────────────────────────────────────────────

  Widget _dropZone() => switch (_fileMode) {
        FileMode.pdf => _pdfZone(),
        FileMode.image => _imageZone(),
      };

  Widget _pdfZone() {
    final has = _pickedPdf != null;
    final c = AppColors.of(context);
    return GestureDetector(
      key: const ValueKey('pdf'),
      onTap: _isAnalyzing ? null : _pickPdf,
      child: AnimatedBuilder(
        animation: _pulseAnim,
        builder: (_, child) =>
            Transform.scale(scale: has ? 1.0 : _pulseAnim.value, child: child),
        child: _shell(
            has,
            has
                ? _previewContent(
                    Icons.picture_as_pdf_rounded,
                    AppTheme.red400,
                    _pickedPdf!.name,
                    '${(_pickedPdf!.size / 1024).toStringAsFixed(1)} KB  •  tap to change')
                : _prompt(Icons.upload_file_outlined, 'Tap to upload PDF',
                    'PDF files only')),
      ),
    );
  }

  Widget _imageZone() {
    final has = _pickedImage != null;
    return GestureDetector(
      key: const ValueKey('image'),
      onTap: _isAnalyzing ? null : _pickImage,
      child: AnimatedBuilder(
        animation: _pulseAnim,
        builder: (_, child) =>
            Transform.scale(scale: has ? 1.0 : _pulseAnim.value, child: child),
        child: has
            ? _imagePrev()
            : _shell(
                false,
                _prompt(Icons.image_outlined, 'Tap to pick image',
                    'JPG, PNG supported')),
      ),
    );
  }

  Widget _imagePrev() {
    final c = AppColors.of(context);
    return Stack(children: [
      ClipRRect(
        borderRadius: BorderRadius.circular(16),
        child: _pickedImageBytes != null
            ? Image.memory(_pickedImageBytes!,
                width: double.infinity, height: 190, fit: BoxFit.cover)
            : Image.file(File(_pickedImage!.path),
                width: double.infinity, height: 190, fit: BoxFit.cover),
      ),
      Positioned.fill(
          child: Container(
        decoration: BoxDecoration(
          borderRadius: BorderRadius.circular(16),
          gradient: LinearGradient(
              begin: Alignment.topCenter,
              end: Alignment.bottomCenter,
              colors: [Colors.transparent, c.navy900.withOpacity(0.78)]),
        ),
      )),
      Positioned(
          bottom: 12,
          left: 14,
          right: 14,
          child: Row(children: [
            Expanded(
                child: Text(_pickedImage!.name,
                    style: GoogleFonts.workSans(
                        fontSize: 12,
                        color: Colors.white,
                        fontWeight: FontWeight.w500),
                    overflow: TextOverflow.ellipsis)),
            GestureDetector(
              onTap: _pickImage,
              child: Container(
                padding:
                    const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
                decoration: BoxDecoration(
                    color: c.brand, borderRadius: BorderRadius.circular(6)),
                child: Text('Change',
                    style: GoogleFonts.workSans(
                        fontSize: 11,
                        color: Theme.of(context).brightness == Brightness.dark
                            ? AppTheme.navy900
                            : Colors.white,
                        fontWeight: FontWeight.w600)),
              ),
            ),
          ])),
    ]);
  }

  Widget _shell(bool has, Widget child) {
    final c = AppColors.of(context);
    final isDark = Theme.of(context).brightness == Brightness.dark;
    // Base surface the tint sits on top of. Blending the brand tint onto a
    // known, opaque base (instead of leaving it as a translucent color left
    // to composite against whatever happens to be behind the widget) keeps
    // the result predictably light-on-dark / dark-on-light, so the text
    // drawn on top never ends up low-contrast against it.
    final baseSurface = isDark ? c.navy800 : Colors.white;
    final selectedFill = Color.alphaBlend(
      c.brand.withOpacity(isDark ? 0.22 : 0.12),
      baseSurface,
    );
    return AnimatedContainer(
      duration: const Duration(milliseconds: 200),
      width: double.infinity,
      constraints: const BoxConstraints(minHeight: 155),
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: has ? selectedFill : baseSurface,
        borderRadius: BorderRadius.circular(AppTheme.radius),
        border: Border.all(
            color: has ? c.brand : c.cardBorder,
            width: has ? AppTheme.borderWidthThick : AppTheme.borderWidth),
        boxShadow:
            AppTheme.hardShadow(has ? c.brand : c.cardBorder, dx: 4, dy: 4),
      ),
      child: child,
    );
  }

  Widget _prompt(IconData icon, String title, String sub) {
    final c = AppColors.of(context);
    return Column(mainAxisAlignment: MainAxisAlignment.center, children: [
      Container(
        width: 52,
        height: 52,
        decoration: BoxDecoration(
            color: c.navy700,
            borderRadius: BorderRadius.circular(14),
            border: Border.all(color: c.cardBorder, width: 2.5)),
        child: Icon(icon, color: c.textTertiary, size: 24),
      ),
      const SizedBox(height: 12),
      Text(title,
          style: GoogleFonts.spaceGrotesk(
              fontSize: 15,
              fontWeight: FontWeight.w600,
              color: c.textSecondary)),
      const SizedBox(height: 4),
      Text(sub,
          style: GoogleFonts.workSans(fontSize: 12, color: c.textTertiary)),
    ]);
  }

  Widget _previewContent(
      IconData icon, Color iconColor, String name, String meta) {
    final c = AppColors.of(context);
    final isDark = Theme.of(context).brightness == Brightness.dark;
    return Column(mainAxisAlignment: MainAxisAlignment.center, children: [
      Stack(clipBehavior: Clip.none, alignment: Alignment.center, children: [
        Container(
          width: 52,
          height: 52,
          decoration: BoxDecoration(
              color: iconColor.withOpacity(0.14),
              borderRadius: BorderRadius.circular(14)),
          child: Icon(icon, color: iconColor, size: 26),
        ),
        Positioned(
          right: -4,
          bottom: -4,
          child: Container(
            width: 20,
            height: 20,
            decoration: BoxDecoration(
              color: c.brand,
              shape: BoxShape.circle,
              border: Border.all(color: c.cardBorder, width: 1.5),
            ),
            child:
                const Icon(Icons.check_rounded, size: 12, color: AppTheme.ink),
          ),
        ),
      ]),
      const SizedBox(height: 12),
      Padding(
        padding: const EdgeInsets.symmetric(horizontal: 16),
        child: Text(name,
            textAlign: TextAlign.center,
            maxLines: 1,
            style: GoogleFonts.spaceGrotesk(
                fontSize: 14,
                fontWeight: FontWeight.w600,
                color: c.textPrimary),
            overflow: TextOverflow.ellipsis),
      ),
      const SizedBox(height: 8),
      Container(
        padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
        decoration: BoxDecoration(
          color: isDark ? c.navy700 : AppTheme.light100,
          borderRadius: BorderRadius.circular(20),
        ),
        child: Text(meta,
            style: GoogleFonts.workSans(
                fontSize: 11,
                fontWeight: FontWeight.w600,
                color: c.textSecondary)),
      ),
    ]);
  }

  // ── Target job field ──────────────────────────────────────────────────────

  Widget _targetJobField() {
    final c = AppColors.of(context);
    final isDark = Theme.of(context).brightness == Brightness.dark;

    return Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
      Text('Job you\'re aiming for',
          style: GoogleFonts.spaceGrotesk(
              fontSize: 13,
              fontWeight: FontWeight.w600,
              color: c.textSecondary)),
      const SizedBox(height: 8),
      LayoutBuilder(builder: (context, constraints) {
        _targetFieldWidth = constraints.maxWidth;
        return RawAutocomplete<String>(
          textEditingController: _targetJobCtrl,
          focusNode: _targetJobFocus,
          optionsBuilder: (TextEditingValue value) {
            final q = value.text.trim().toLowerCase();
            if (q.isEmpty || _jobTitleSuggestions.isEmpty) {
              return const Iterable<String>.empty();
            }
            return _jobTitleSuggestions
                .where((t) => t.toLowerCase().contains(q))
                .take(6);
          },
          fieldViewBuilder: (context, controller, focusNode, onSubmitted) {
            return TextFormField(
              controller: controller,
              focusNode: focusNode,
              enabled: !_isAnalyzing,
              style: GoogleFonts.workSans(fontSize: 14, color: c.textPrimary),
              decoration: const InputDecoration(
                hintText: 'e.g. Data Scientist, Software Engineer',
                prefixIcon: Icon(Icons.work_outline_rounded,
                    color: AppTheme.textTertiary, size: 18),
              ),
              validator: (v) => (v == null || v.trim().isEmpty)
                  ? 'Enter the target job title'
                  : null,
              onFieldSubmitted: (_) => onSubmitted(),
            );
          },
          optionsViewBuilder: (context, onSelected, options) {
            return Align(
              alignment: Alignment.topLeft,
              child: Material(
                color: Colors.transparent,
                child: Container(
                  width: _targetFieldWidth,
                  margin: const EdgeInsets.only(top: 6),
                  constraints: const BoxConstraints(maxHeight: 230),
                  decoration: BoxDecoration(
                    color: isDark ? c.navy800 : Colors.white,
                    borderRadius: BorderRadius.circular(12),
                    border: Border.all(color: c.cardBorder, width: 2.5),
                    boxShadow: AppTheme.hardShadow(c.cardBorder, dx: 4, dy: 4),
                  ),
                  child: ListView.separated(
                    padding: const EdgeInsets.symmetric(vertical: 4),
                    shrinkWrap: true,
                    itemCount: options.length,
                    separatorBuilder: (_, __) =>
                        Divider(height: 1, color: c.divider),
                    itemBuilder: (context, index) {
                      final opt = options.elementAt(index);
                      return InkWell(
                        onTap: () => onSelected(opt),
                        child: Padding(
                          padding: const EdgeInsets.symmetric(
                              horizontal: 14, vertical: 12),
                          child: Row(children: [
                            Icon(Icons.history_rounded,
                                size: 14, color: c.textTertiary),
                            const SizedBox(width: 10),
                            Expanded(
                              child: Text(opt,
                                  style: GoogleFonts.workSans(
                                      fontSize: 13, color: c.textPrimary),
                                  overflow: TextOverflow.ellipsis),
                            ),
                          ]),
                        ),
                      );
                    },
                  ),
                ),
              ),
            );
          },
        );
      }),
    ]);
  }

  // ── Error / loading / button ──────────────────────────────────────────────

  Widget _errorBox() => Container(
        margin: const EdgeInsets.only(bottom: 16),
        padding: const EdgeInsets.all(14),
        decoration: BoxDecoration(
          color: AppTheme.red500.withOpacity(0.08),
          borderRadius: BorderRadius.circular(12),
          border:
              Border.all(color: AppTheme.red500.withOpacity(0.3), width: 2.5),
        ),
        child: Row(crossAxisAlignment: CrossAxisAlignment.start, children: [
          const Icon(Icons.error_outline_rounded,
              color: AppTheme.red400, size: 16),
          const SizedBox(width: 10),
          Expanded(
              child: Text(_error!,
                  style: GoogleFonts.workSans(
                      fontSize: 13, color: AppTheme.red400))),
        ]),
      );

  Widget _analyzingState() {
    final c = AppColors.of(context);
    return Column(children: [
      LinearProgressIndicator(backgroundColor: c.navy700, color: c.brand),
      const SizedBox(height: 12),
      Text(_statusMessage,
          style: GoogleFonts.workSans(fontSize: 13, color: c.textSecondary),
          textAlign: TextAlign.center),
    ]);
  }

  Widget _analyzeBtn() {
    final isTargeted = _analysisType == AnalysisType.targeted;
    return SizedBox(
      width: double.infinity,
      child: ElevatedButton.icon(
        onPressed: _analyze,
        icon: Icon(
            isTargeted ? Icons.gps_fixed_rounded : Icons.grid_view_rounded,
            size: 18),
        label: Text(
            isTargeted ? 'Analyze — Target Career' : 'Analyze — Full Analysis'),
      ),
    );
  }
}
