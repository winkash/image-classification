import argparse
from tempfile import mkdtemp
from logging import getLogger

from affine.external.crawl.image_crawler_by_keyword import GoogleImageCrawler, \
    save_images

logger = getLogger('affine.logo.data_grabber')


def main(file_path, download_dir):
    with open(file_path, 'r') as f:
        logo_names = list(set(f.readlines()))

    for lname in logo_names:
        lname = lname.strip()
        lname = lname.replace(' ', '-')
        query = lname + ' ' + 'logo'
        gic = GoogleImageCrawler(query, 5)
        urls = gic.grab_image_urls()
        logger.info("Got %s urls for %s", len(urls), lname)
        save_images(urls, lname, download_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('logo_names_filepath',
                        help="A file containing names of all logos for which \
                        we need to acquire data. 1 name per line")
    parser.add_argument('download_dir',
                        help="the firectory where images should be downloaded")
    args = parser.parse_args()
    main(args.logo_names_filepath, args.download_dir)
