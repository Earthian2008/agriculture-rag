from html_to_text import extract_text


PAGES = [

    {
        "name": "season_and_varieties",
        "url": "http://www.agritech.tnau.ac.in/agriculture/agri_seasonandvarieties_rice.html"
    },

    {
        "name": "seed_treatment",
        "url": "http://www.agritech.tnau.ac.in/agriculture/agri_Seed%20Treatment_Rice.html"
    },

    {
        "name": "nursery_management",
        "url": "http://www.agritech.tnau.ac.in/agriculture/Nursery%20Sowing%20__%20An%20Introduction1.html"
    },

    {
        "name": "main_field_preparation",
        "url": "http://www.agritech.tnau.ac.in/agriculture/agri_Main_Field_preparation_Rice%201.html"
    },

    {
        "name": "irrigation_management",
        "url": "http://www.agritech.tnau.ac.in/agriculture/agri_irrigationmgt_cereals_rice1.html"
    },

    {
        "name": "organic_nutrients",
        "url": "http://www.agritech.tnau.ac.in/agriculture/agri_Nutrient_mainfeild_%20Organic_biofertilizers_Rice.html"
    },

    {
        "name": "biofertilizers",
        "url": "http://www.agritech.tnau.ac.in/agriculture/agri_Nutrient_mainfeild_biofertilizers_Rice.html"
    },

    {
        "name": "inorganic_nutrients",
        "url": "http://www.agritech.tnau.ac.in/agriculture/agri_main_feild_%20InOrganic_biofertilizers_Rice.html"
    },

    {
        "name": "weed_management",
        "url": "http://www.agritech.tnau.ac.in/agriculture/agri_weedmgt_rice.html"
    }

]


OUTPUT_DIRECTORY = "data/rice"


def main():

    print("=" * 60)
    print("TNAU RICE KNOWLEDGE INGESTION")
    print("=" * 60)

    successful = 0
    failed = 0

    for page in PAGES:

        output_file = (
            f"{OUTPUT_DIRECTORY}/{page['name']}.txt"
        )

        try:

            extract_text(
                page["url"],
                output_file
            )

            successful += 1

        except Exception as error:

            print(
                f"FAILED: {page['name']}"
            )

            print(
                f"Reason: {error}"
            )

            failed += 1

    print()
    print("=" * 60)
    print("INGESTION COMPLETE")
    print("=" * 60)

    print(
        f"Successful: {successful}"
    )

    print(
        f"Failed: {failed}"
    )


if __name__ == "__main__":
    main()
