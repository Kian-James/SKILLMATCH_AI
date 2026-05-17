// lib/screens/camera_scan_screen.dart
// Mode 3: live camera scan → ML Kit OCR → extracted text preview

import 'dart:io';
import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:google_mlkit_text_recognition/google_mlkit_text_recognition.dart';
import '../theme.dart';
import '../widgets/common_widgets.dart';

class CameraScanScreen extends StatefulWidget {
  const CameraScanScreen({super.key});

  @override
  State<CameraScanScreen> createState() => _CameraScanScreenState();
}

class _CameraScanScreenState extends State<CameraScanScreen>
    with WidgetsBindingObserver {
  CameraController? _ctrl;
  List<CameraDescription> _cameras = [];
  bool _isInitialized = false;
  bool _isCapturing = false;
  bool _isProcessing = false;

  // Multi-page scanning
  final List<File> _capturedPages = [];
  final List<String> _extractedTexts = [];
  String _previewText = '';
  String? _error;

  final _textRecognizer = TextRecognizer(script: TextRecognitionScript.latin);

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    _initCamera();
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    _ctrl?.dispose();
    _textRecognizer.close();
    super.dispose();
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    if (_ctrl == null || !_ctrl!.value.isInitialized) return;
    if (state == AppLifecycleState.inactive) {
      _ctrl?.dispose();
    } else if (state == AppLifecycleState.resumed) {
      _initCamera();
    }
  }

  Future<void> _initCamera() async {
    try {
      _cameras = await availableCameras();
      if (_cameras.isEmpty) {
        setState(() => _error = 'No camera found on this device.');
        return;
      }

      final back = _cameras.firstWhere(
        (c) => c.lensDirection == CameraLensDirection.back,
        orElse: () => _cameras.first,
      );

      _ctrl = CameraController(
        back,
        ResolutionPreset.veryHigh, // high res for OCR accuracy
        enableAudio: false,
        imageFormatGroup: ImageFormatGroup.jpeg,
      );

      await _ctrl!.initialize();
      if (mounted) setState(() => _isInitialized = true);
    } catch (e) {
      setState(() => _error = 'Camera error: $e');
    }
  }

  Future<void> _captureAndOcr() async {
    if (_ctrl == null || !_ctrl!.value.isInitialized || _isCapturing) return;

    setState(() {
      _isCapturing = true;
      _isProcessing = true;
      _error = null;
    });

    try {
      // Capture
      final xFile = await _ctrl!.takePicture();
      final file = File(xFile.path);
      _capturedPages.add(file);

      // Run ML Kit OCR
      final inputImage = InputImage.fromFile(file);
      final recognized = await _textRecognizer.processImage(inputImage);

      final pageText = recognized.blocks
          .map((b) => b.lines.map((l) => l.text).join('\n'))
          .join('\n\n');

      _extractedTexts.add(pageText);

      // Merge all pages
      _previewText = _extractedTexts.join('\n\n--- Page break ---\n\n');

      setState(() {});
    } catch (e) {
      setState(() => _error = 'OCR failed: $e');
    } finally {
      setState(() {
        _isCapturing = false;
        _isProcessing = false;
      });
    }
  }

  void _removePage(int index) {
    setState(() {
      _capturedPages.removeAt(index);
      _extractedTexts.removeAt(index);
      _previewText = _extractedTexts.join('\n\n--- Page break ---\n\n');
    });
  }

  void _confirmAndReturn() {
    if (_extractedTexts.isEmpty) {
      setState(() => _error = 'Scan at least one page first.');
      return;
    }
    Navigator.pop(context, {
      'text': _previewText,
      'page_count': _capturedPages.length,
      'files': _capturedPages,
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.black,
      appBar: AppBar(
        backgroundColor: Colors.black,
        title: Text(
          'Scan Resume  •  ${_capturedPages.length} page${_capturedPages.length != 1 ? 's' : ''}',
          style: GoogleFonts.spaceGrotesk(
              color: Colors.white, fontWeight: FontWeight.w600),
        ),
        iconTheme: const IconThemeData(color: Colors.white),
        actions: [
          if (_capturedPages.isNotEmpty)
            TextButton(
              onPressed: _confirmAndReturn,
              child: Text(
                'Use (${_capturedPages.length})',
                style: GoogleFonts.spaceGrotesk(
                    color: AppTheme.teal500, fontWeight: FontWeight.w700),
              ),
            ),
        ],
      ),
      body: Column(
        children: [
          // Camera viewfinder
          Expanded(
            flex: 3,
            child: Stack(
              fit: StackFit.expand,
              children: [
                if (_isInitialized && _ctrl != null)
                  _buildViewfinder()
                else if (_error != null)
                  _buildCameraError()
                else
                  const Center(child: CircularProgressIndicator(color: AppTheme.teal500)),

                // Scan overlay frame
                if (_isInitialized) _buildScanFrame(),

                // Processing overlay
                if (_isProcessing)
                  Container(
                    color: Colors.black54,
                    child: Center(
                      child: Column(
                        mainAxisSize: MainAxisSize.min,
                        children: [
                          const CircularProgressIndicator(color: AppTheme.teal500),
                          const SizedBox(height: 16),
                          Text('Extracting text…',
                              style: GoogleFonts.inter(
                                  color: Colors.white, fontSize: 14)),
                        ],
                      ),
                    ),
                  ),
              ],
            ),
          ),

          // Bottom controls
          Container(
            color: const Color(0xFF0A0A0A),
            padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 16),
            child: Column(
              children: [
                if (_error != null) _buildError(),

                // Page thumbnails
                if (_capturedPages.isNotEmpty) ...[
                  _buildPageThumbnails(),
                  const SizedBox(height: 12),
                ],

                // Capture row
                Row(
                  mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                  children: [
                    // OCR text preview
                    _circleButton(
                      icon: Icons.text_snippet_outlined,
                      color: _extractedTexts.isEmpty
                          ? Colors.grey
                          : AppTheme.amber500,
                      label: 'Preview',
                      onTap: _extractedTexts.isEmpty
                          ? null
                          : () => _showTextPreview(context),
                    ),

                    // Capture button
                    GestureDetector(
                      onTap: _isCapturing ? null : _captureAndOcr,
                      child: AnimatedContainer(
                        duration: const Duration(milliseconds: 150),
                        width: 72,
                        height: 72,
                        decoration: BoxDecoration(
                          shape: BoxShape.circle,
                          border: Border.all(
                              color: AppTheme.teal500,
                              width: _isCapturing ? 3 : 4),
                          color: _isCapturing
                              ? AppTheme.teal500.withOpacity(0.2)
                              : Colors.transparent,
                        ),
                        child: Center(
                          child: Container(
                            width: 54,
                            height: 54,
                            decoration: const BoxDecoration(
                              shape: BoxShape.circle,
                              color: AppTheme.teal500,
                            ),
                            child: _isCapturing
                                ? const Padding(
                                    padding: EdgeInsets.all(16),
                                    child: CircularProgressIndicator(
                                        color: AppTheme.navy900, strokeWidth: 2),
                                  )
                                : const Icon(Icons.camera_alt_rounded,
                                    color: AppTheme.navy900, size: 24),
                          ),
                        ),
                      ),
                    ),

                    // Confirm / done
                    _circleButton(
                      icon: Icons.check_rounded,
                      color: _capturedPages.isEmpty
                          ? Colors.grey
                          : AppTheme.green500,
                      label: 'Done',
                      onTap: _capturedPages.isEmpty ? null : _confirmAndReturn,
                    ),
                  ],
                ),
                const SizedBox(height: 6),
                Text(
                  'Tap shutter to scan each page  •  Scan multiple pages',
                  style: GoogleFonts.inter(
                      fontSize: 11, color: Colors.grey[600]),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildViewfinder() {
    return ClipRect(
      child: OverflowBox(
        alignment: Alignment.center,
        child: FittedBox(
          fit: BoxFit.cover,
          child: SizedBox(
            width: _ctrl!.value.previewSize!.height,
            height: _ctrl!.value.previewSize!.width,
            child: CameraPreview(_ctrl!),
          ),
        ),
      ),
    );
  }

  Widget _buildScanFrame() {
    return Center(
      child: Container(
        width: MediaQuery.of(context).size.width * 0.82,
        height: MediaQuery.of(context).size.height * 0.42,
        decoration: BoxDecoration(
          borderRadius: BorderRadius.circular(12),
          border: Border.all(color: AppTheme.teal500.withOpacity(0.8), width: 2),
        ),
        child: Stack(
          children: [
            // Corner accents
            _corner(Alignment.topLeft),
            _corner(Alignment.topRight),
            _corner(Alignment.bottomLeft),
            _corner(Alignment.bottomRight),
          ],
        ),
      ),
    );
  }

  Widget _corner(Alignment alignment) {
    final isTop = alignment == Alignment.topLeft || alignment == Alignment.topRight;
    final isLeft = alignment == Alignment.topLeft || alignment == Alignment.bottomLeft;

    return Align(
      alignment: alignment,
      child: Container(
        width: 24,
        height: 24,
        decoration: BoxDecoration(
          border: Border(
            top: isTop ? const BorderSide(color: AppTheme.teal500, width: 3) : BorderSide.none,
            bottom: !isTop ? const BorderSide(color: AppTheme.teal500, width: 3) : BorderSide.none,
            left: isLeft ? const BorderSide(color: AppTheme.teal500, width: 3) : BorderSide.none,
            right: !isLeft ? const BorderSide(color: AppTheme.teal500, width: 3) : BorderSide.none,
          ),
        ),
      ),
    );
  }

  Widget _buildPageThumbnails() {
    return SizedBox(
      height: 64,
      child: ListView.separated(
        scrollDirection: Axis.horizontal,
        itemCount: _capturedPages.length,
        separatorBuilder: (_, __) => const SizedBox(width: 8),
        itemBuilder: (_, i) => Stack(
          children: [
            ClipRRect(
              borderRadius: BorderRadius.circular(8),
              child: Image.file(
                _capturedPages[i],
                width: 46,
                height: 64,
                fit: BoxFit.cover,
              ),
            ),
            Positioned(
              top: 2,
              right: 2,
              child: GestureDetector(
                onTap: () => _removePage(i),
                child: Container(
                  width: 16,
                  height: 16,
                  decoration: const BoxDecoration(
                    color: AppTheme.red500,
                    shape: BoxShape.circle,
                  ),
                  child: const Icon(Icons.close, size: 10, color: Colors.white),
                ),
              ),
            ),
            Positioned(
              bottom: 2,
              left: 2,
              child: Container(
                padding: const EdgeInsets.symmetric(horizontal: 4, vertical: 1),
                decoration: BoxDecoration(
                  color: Colors.black54,
                  borderRadius: BorderRadius.circular(4),
                ),
                child: Text('${i + 1}',
                    style: const TextStyle(color: Colors.white, fontSize: 9)),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildError() {
    return Container(
      margin: const EdgeInsets.only(bottom: 10),
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
      decoration: BoxDecoration(
        color: AppTheme.red500.withOpacity(0.15),
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: AppTheme.red500.withOpacity(0.4)),
      ),
      child: Text(_error!,
          style: GoogleFonts.inter(fontSize: 12, color: AppTheme.red400)),
    );
  }

  Widget _buildCameraError() {
    return Center(
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          const Icon(Icons.camera_outlined, color: AppTheme.textTertiary, size: 48),
          const SizedBox(height: 12),
          Text(_error ?? 'Camera unavailable',
              style: GoogleFonts.inter(color: AppTheme.textSecondary),
              textAlign: TextAlign.center),
        ],
      ),
    );
  }

  Widget _circleButton({
    required IconData icon,
    required Color color,
    required String label,
    required VoidCallback? onTap,
  }) {
    return GestureDetector(
      onTap: onTap,
      child: Column(
        children: [
          Container(
            width: 50,
            height: 50,
            decoration: BoxDecoration(
              shape: BoxShape.circle,
              color: color.withOpacity(0.12),
              border: Border.all(color: color.withOpacity(0.3)),
            ),
            child: Icon(icon, color: color, size: 22),
          ),
          const SizedBox(height: 4),
          Text(label,
              style: GoogleFonts.inter(fontSize: 10, color: Colors.grey[500])),
        ],
      ),
    );
  }

  void _showTextPreview(BuildContext context) {
    showModalBottomSheet(
      context: context,
      backgroundColor: AppTheme.navy800,
      isScrollControlled: true,
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(20)),
      ),
      builder: (_) => DraggableScrollableSheet(
        initialChildSize: 0.65,
        maxChildSize: 0.92,
        minChildSize: 0.35,
        expand: false,
        builder: (_, scrollCtrl) => Column(
          children: [
            Container(
              margin: const EdgeInsets.symmetric(vertical: 10),
              width: 36,
              height: 4,
              decoration: BoxDecoration(
                color: AppTheme.navy500,
                borderRadius: BorderRadius.circular(2),
              ),
            ),
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 4),
              child: Row(
                children: [
                  Text('Extracted Text',
                      style: GoogleFonts.spaceGrotesk(
                          color: AppTheme.textPrimary,
                          fontWeight: FontWeight.w600,
                          fontSize: 16)),
                  const Spacer(),
                  TealBadge(text: '${_capturedPages.length} page${_capturedPages.length != 1 ? 's' : ''}'),
                ],
              ),
            ),
            const Divider(color: AppTheme.divider),
            Expanded(
              child: SingleChildScrollView(
                controller: scrollCtrl,
                padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 12),
                child: Text(
                  _previewText.isEmpty ? 'No text extracted yet.' : _previewText,
                  style: GoogleFonts.jetBrainsMono(
                    fontSize: 12,
                    color: AppTheme.textSecondary,
                    height: 1.7,
                  ),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
