#!/usr/bin/env python3
"""
RVC environment setup and management CLI.
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.rvc_environment import (
    setup_rvc_environment,
    validate_rvc_environment,
    cleanup_rvc_environment,
    get_rvc_environment_info,
    check_rvc_environment_exists,
    download_pretrained_models
)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Manage RVC environment for AutoVid"
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Setup command
    setup_parser = subparsers.add_parser('setup', help='Setup RVC environment')
    setup_parser.add_argument('--force', action='store_true', 
                             help='Force recreation of environment')
    
    # Status command
    subparsers.add_parser('status', help='Show RVC environment status')
    
    # Validate command
    subparsers.add_parser('validate', help='Validate RVC environment')
    
    # Download command
    subparsers.add_parser('download', help='Download pretrained models only')
    
    # Cleanup command
    subparsers.add_parser('cleanup', help='Remove RVC environment')
    
    return parser.parse_args()


def cmd_setup(force=False):
    """Setup RVC environment."""
    if force and check_rvc_environment_exists():
        print("Forcing cleanup of existing environment...")
        if not cleanup_rvc_environment():
            print("Failed to cleanup existing environment")
            return False
    
    print("Setting up RVC environment...")
    success = setup_rvc_environment()
    
    if success:
        print("✅ RVC environment setup complete!")
        cmd_status()
    else:
        print("❌ RVC environment setup failed!")
    
    return success


def cmd_status():
    """Show RVC environment status."""
    print("\n📊 RVC Environment Status:")
    print("=" * 40)
    
    info = get_rvc_environment_info()
    
    # Environment status
    if info["environment_exists"]:
        print("🟢 Conda Environment: EXISTS")
        print(f"   Path: {info['environment_path']}")
    else:
        print("🔴 Conda Environment: MISSING")
    
    # RVC script status
    if info["rvc_script_exists"]:
        print("🟢 RVC Script: EXISTS")
    else:
        print("🔴 RVC Script: MISSING")
    
    # Pretrained models status
    print("\n📦 Pretrained Models:")
    for model_name, model_info in info["pretrained_models"].items():
        if model_info["exists"]:
            size_mb = model_info["size"] / (1024 * 1024)
            print(f"🟢 {model_name}: EXISTS ({size_mb:.1f} MB)")
        else:
            print(f"🔴 {model_name}: MISSING")
    
    # Overall status
    all_ready = (
        info["environment_exists"] and 
        info["rvc_script_exists"] and
        info["pretrained_models"]["hubert_base.pt"]["exists"]
    )
    
    print(f"\n🎯 Overall Status: {'READY' if all_ready else 'NOT READY'}")


def cmd_validate():
    """Validate RVC environment."""
    print("🔍 Validating RVC environment...")
    
    is_valid, error_msg = validate_rvc_environment()
    
    if is_valid:
        print("✅ RVC environment validation successful!")
        print("   Environment is ready for use.")
    else:
        print("❌ RVC environment validation failed!")
        print(f"   Error: {error_msg}")
        print("\n💡 Try running: python cli/rvc_setup.py setup")
    
    return is_valid


def cmd_download():
    """Download pretrained models only."""
    print("📥 Downloading pretrained models...")
    
    success = download_pretrained_models()
    
    if success:
        print("✅ Pretrained models downloaded successfully!")
    else:
        print("❌ Failed to download pretrained models!")
    
    return success


def cmd_cleanup():
    """Remove RVC environment."""
    print("🗑️  Removing RVC environment...")
    
    # Confirm with user
    response = input("Are you sure you want to remove the RVC environment? (y/N): ")
    if response.lower() != 'y':
        print("Cancelled.")
        return False
    
    success = cleanup_rvc_environment()
    
    if success:
        print("✅ RVC environment removed successfully!")
    else:
        print("❌ Failed to remove RVC environment!")
    
    return success


def main():
    """Main execution function."""
    args = parse_args()
    
    if not args.command:
        print("No command specified. Use --help for available commands.")
        sys.exit(1)
    
    try:
        if args.command == 'setup':
            success = cmd_setup(force=args.force)
        elif args.command == 'status':
            cmd_status()
            success = True
        elif args.command == 'validate':
            success = cmd_validate()
        elif args.command == 'download':
            success = cmd_download()
        elif args.command == 'cleanup':
            success = cmd_cleanup()
        else:
            print(f"Unknown command: {args.command}")
            sys.exit(1)
        
        if not success:
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()